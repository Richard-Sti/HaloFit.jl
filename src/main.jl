
mutable struct Halo{T<:Real}
    pos::Matrix{T}
    vel::Matrix{T}
    mass::Vector{T}
    boxsize::T
    cm::Vector{T}
    dist::Vector{T}
    is_sorted::Bool
end

Base.length(halo::Halo) = size(halo.pos, 1)


function Halo(pos::Matrix{U}, vel::Matrix{U}, mass::Vector{U}, boxsize::V) where {U <: Real, V <: Real}
    return Halo(pos,
                vel,
                mass,
                convert(U, boxsize),
                Vector{U}(undef, 3),
                Vector{U}(undef, size(pos, 1)),
                false
                )
end


function meansigma(x::Vector{T}) where T <: Real
    npoints = length(x)

    μ = sum(x) / npoints
    σ = T(0.)

    @inbounds @fastmath for i in 1:npoints
        σ += (x[i] - μ)^2
    end

    σ = sqrt(σ / npoints)
    return μ, σ
end


function periodic_distance!(halo::Halo{T}) where T <: Real
    periodic_distance!(halo.dist, halo.pos, halo.cm, halo.boxsize)

    sorted_indices = sortperm(halo.dist)

    halo.pos .= halo.pos[sorted_indices, :]
    halo.vel .= halo.vel[sorted_indices, :]

    permute!(halo.mass, sorted_indices)
    permute!(halo.dist, sorted_indices)

    halo.is_sorted = true

    return halo
end


function periodic_distance!(dist::Vector{T}, points::Matrix{T}, reference::Vector{T}, boxsize::T) where T <: Real
    halfbox = T(boxsize / 2.0)

    @inbounds for i in 1:size(points, 1)
        dist_sq = T(0.0)
        @simd for j in 1:3
            dist_1d = abs(points[i, j] - reference[j])
            dist_1d = dist_1d > halfbox ? boxsize - dist_1d : dist_1d
            dist_sq += dist_1d^2
        end

        dist[i] = sqrt(dist_sq)
    end

    return dist
end


function center_of_mass!(cm::Vector{T}, sin_inv_points::Matrix{T}, cos_inv_points::Matrix{T},
                         mass::Vector{T}, boxsize::T; mask::Union{Vector{Bool}, Nothing}=nothing) where T <: Real
    T2π = T(2 * π)

    @inbounds @fastmath for i in 1:3
        cm_i_real = T(0.0)
        cm_i_imag = T(0.0)
        @inbounds @fastmath for j in 1:length(mass)
            (mask === nothing || mask[j]) || continue

            m = mass[j]
            cm_i_real += m * cos_inv_points[j, i]
            cm_i_imag += m * sin_inv_points[j, i]
        end

        cm_i = atan(cm_i_imag, cm_i_real) * boxsize / T2π
        cm_i = cm_i < 0 ? cm_i + boxsize : cm_i

        cm[i] = cm_i
    end

    return cm
end

function center_of_mass!(cm::Vector{T}, points::Matrix{T}, mass::Vector{T}, boxsize::T) where T <: Real
    center_of_mass!(
        cm,
        sin.(T(2 * π) .* points ./ boxsize),
        cos.(T(2 * π) .* points ./ boxsize),
        mass,
        boxsize
        )
end


function shrinking_sphere_cm!(halo::Halo{T}; npart_min::Int=50, shrink_factor::Real=0.95) where T <: Real
    npoints = length(halo)
    shrink_factor = T(shrink_factor)

    sin_inv_points = sin.(T(2 * π) .* halo.pos ./ halo.boxsize)
    cos_inv_points = cos.(T(2 * π) .* halo.pos ./ halo.boxsize)

    # Initial guess
    center_of_mass!(halo.cm, sin_inv_points, cos_inv_points, halo.mass, halo.boxsize)

    rad = nothing
    within_rad = fill(true, npoints)
    while true
         periodic_distance!(halo.dist, halo.pos, halo.cm, halo.boxsize)

         if rad === nothing
             μ, σ = meansigma(halo.dist)
             rad = μ + T(1.5) * σ
         end

        @inbounds for i in 1:npoints
            within_rad[i] = halo.dist[i] <= rad
        end

        center_of_mass!(halo.cm, sin_inv_points, cos_inv_points, halo.mass, halo.boxsize;
                        mask=within_rad)

        sum(within_rad) < npart_min && return periodic_distance!(halo)
        rad *= shrink_factor
    end
end


function spherical_overdensity_mass(halo::Halo, ρtarget::T) where T <: Real

    @assert halo.is_sorted "Halo must be sorted by distance from its CM"

    ρ = cumsum(halo.mass)
    totmass = ρ[end]
    ρ ./= (T(4 / 3 * π) .* halo.dist.^3)
    ρ ./= ρtarget

    j = find_first_below_threshold(ρ, T(1))

    if j === nothing
        return NaN, NaN
    end

    i = j - 1

    rad = @inbounds (halo.dist[j] - halo.dist[i]) * (1 - ρ[i]) / (ρ[j] - ρ[i]) + halo.dist[i]
    mass_in_rad = T(4 / 3 *  π) * ρtarget .* rad.^3

    if mass_in_rad > totmass
        return T(NaN), T(NaN)
    end

    return mass_in_rad, rad
end


function find_first_below_threshold(x::Vector{<:T}, threshold::T) where T <: Real
    @inbounds for i in 2:length(x)
        if x[i] < threshold
            return i
        end
    end

    return nothing
end


function find_first_above_threshold(x::Vector{<:T}, threshold::T) where T <: Real
    @inbounds for i in 2:length(x)
        if x[i] > threshold
            return i
        end
    end

    return nothing
end


function angular_momentum(halo::Halo{T}, rad::T) where T <: Real
    @assert halo.is_sorted "Halo must be sorted by distance from its CM"
    pos = copy(halo.pos)
    vel = copy(halo.vel)

    shift_pos_to_center_of_box!(pos, halo.cm, halo.boxsize)
    shift_vel_to_cm_frame!(vel, halo.mass)

    angmom = zeros(T, 3)
    imax = find_first_above_threshold(halo.dist, rad)

    if imax === nothing
        angmom *= NaN
        return angmom
    end

    @inbounds @fastmath for i in 1:(imax - 1)
        m = halo.mass[i]
        angmom[1] += m * (pos[i, 2] * vel[i, 3] - pos[i, 3] * vel[i, 2])
        angmom[2] += m * (pos[i, 3] * vel[i, 1] - pos[i, 1] * vel[i, 3])
        angmom[3] += m * (pos[i, 1] * vel[i, 2] - pos[i, 2] * vel[i, 1])
    end

    return angmom
end


function λbullock(angmom::Vector{<:T}, mass::T, rad::T) where T <: Real
    G = T(4.300917270069976e-09)  # G in (Msun / h)^-1 (Mpc / h) (km / s)^2
    return @fastmath sqrt(sum(angmom.^2)) / sqrt(2 * G * mass^3 * rad)
end


function shift_pos_to_center_of_box!(points::Matrix{T}, cm::Vector{T}, boxsize::T) where T <: Real
    hw = boxsize / 2
    @inbounds for i in 1:size(points, 1)
        @inbounds @fastmath for j in 1:3
            points[i, j] = (points[i, j] + hw - cm[j]) % boxsize - hw
        end
    end
end


function shift_vel_to_cm_frame!(vel::Matrix{T}, mass::Vector{T}) where T <: Real
    npoints = size(vel, 1)
    totmass = sum(mass)

    @fastmath @inbounds for i in 1:3
        vel_i = 0.
        @fastmath @inbounds for j in 1:npoints
            vel_i += mass[j] * vel[j, i]
        end

        vel_i /= totmass
        vel[:, i] .-= vel_i
    end
end


function ρcrit0(h::T) where T <: Real
    return 2.77536627e+11 * h^2
end
