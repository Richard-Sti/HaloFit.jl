# Copyright (C) 2023 Richard Stiskalek
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
using LinearAlgebra, LoopVectorization, Optim

mutable struct Halo{T<:Real}
    pos::Matrix{T}
    vel::Matrix{T}
    mass::Vector{T}
    potential::Vector{T}
    boxsize::T
    cm::Vector{T}
    dist::Vector{T}
    is_sorted::Bool
end


Base.length(halo::Halo) = size(halo.pos, 1)
ρcrit0(h::Real) = 2.77536627e+11 * h^2  # [ρcrit0] = Msun / Mpc^3
Base.show(io::IO, halo::Halo) = print(io, "Halo($(length(halo)) particles)")


function Halo(pos::Matrix{U}, vel::Matrix{U}, mass::Vector{U}, potential::Vector{U}, boxsize::V) where {U <: Real, V <: Real}
    return Halo(pos,
                vel,
                mass,
                potential,
                convert(U, boxsize),
                Vector{U}(undef, 3),
                Vector{U}(undef, size(pos, 1)),
                false
                )
end


function Halo(pos::Matrix{U}, vel::Matrix{U}, mass::Vector{U}, boxsize::V) where {U <: Real, V <: Real}
    return Halo(pos,
                vel,
                mass,
                Vector{U}(undef, size(pos, 1)),
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


"""
    periodic_distance!(dist::Vector{T}, points::Matrix{T}, reference::Vector{T}, boxsize::T) where T <: Real

Compute the periodic distance between a set of `points` and a `reference` point within a box of size `boxsize`.

# Argumets
- `dist`: A pre-allocated vector to store the resulting distances.
- `points`: A matrix where each row is a point in 3D space.
- `reference`: A vector representing the reference point in 3D space.
- `boxsize`: The size of the periodic box.
"""
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


"""
    center_of_mass!(cm::Vector{T}, sin_inv_points::Matrix{T}, cos_inv_points::Matrix{T},
                    mass::Vector{T}, boxsize::T; mask::Union{Vector{Bool}, Nothing}=nothing) where T <: Real

Compute the center of mass for a set of points.

# Arguments
- `cm`: Vector to store center of mass coordinates (3D).
- `sin_inv_points`, `cos_inv_points`: (N x 3) matrices of sine and cosine of inverse point
   coordinates (2π * points / boxsize).
- `mass`: Vector of masses for each point.
- `boxsize`: Size of the periodic box.
- `mask` (optional): Boolean mask to consider subset of points.
"""
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


"""
    shrinking_sphere_cm!(halo::Halo{T}; npart_min::Int=50, shrink_factor::Real=0.975) where T <: Real

Calculate the center of mass by iteratively refining the center of mass of a `halo` using the
shrinking sphere method.

# Arguments
- `halo`: Halo object containing positions, masses, and other halo properties.
- `npart_min` (optional): Minimum number of particles in the sphere. Default is 50.
- `shrink_factor` (optional): Factor by which the sphere's radius is reduced in each iteration. Default is 0.975.
"""
function shrinking_sphere_cm!(halo::Halo{T}; npart_min::Int=50, shrink_factor::Real=0.975) where T <: Real
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


"""
    spherical_overdensity_mass(halo::Halo, ρtarget::T) where T <: Real

Determine the mass and radius enclosing a target overdensity `ρtarget` for a given `halo`.

# Arguments
- `halo`: Halo object sorted by distance from its center of mass.
- `ρtarget`: Target overdensity value.

# Returns
- `mass_in_rad`: Mass within the computed radius.
- `rad`: Radius enclosing the target overdensity.
"""
function spherical_overdensity_mass(halo::Halo{T}, ρtarget::T) where T <: Real
    @assert halo.is_sorted "Halo must be sorted by distance from its CM"

    ρ = cumsum(halo.mass)
    totmass = ρ[end]
    ρ ./= (T(4 / 3 * π) .* halo.dist.^3)
    ρ ./= ρtarget

    if ρ[2] > 1
        j = find_first_below_threshold(ρ, T(1))
    else
        j = find_first_above_threshold(ρ, T(1))
    end

    if j === nothing
        return T(NaN), T(NaN)
    end

    i = j - 1

    if @inbounds !((ρ[i] > 1 && ρ[j] < 1) || (ρ[i] < 1 && ρ[j] > 1))
        return T(NaN), T(NaN)
    end

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


"""
    function angular_momentum(halo::Halo{T}, rad::T) where T <: Real

Calculate the angular momentum within a given radius `rad` for a specified `halo`.

# Arguments
- `halo`: Halo object sorted by distance from its center of mass.
- `rad`: Radius within which to compute the angular momentum.

# Returns
- A 3D vector representing the computed angular momentum.
"""
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

    imax -= 1

    @inbounds @fastmath for i in 1:(imax)
        m = halo.mass[i]
        angmom[1] += m * (pos[i, 2] * vel[i, 3] - pos[i, 3] * vel[i, 2])
        angmom[2] += m * (pos[i, 3] * vel[i, 1] - pos[i, 1] * vel[i, 3])
        angmom[3] += m * (pos[i, 1] * vel[i, 2] - pos[i, 2] * vel[i, 1])
    end

    return angmom
end


"""
    λbullock(angmom::Vector{<:T}, mass::T, rad::T) where T <: Real

Compute the Bullock spin parameter for a given angular momentum, mass, and radius.

# Arguments
- `angmom`: 3D vector of angular momentum.
- `mass`: Total mass.
- `rad`: Radius.

# Returns
- The computed Bullock spin parameter.
"""
function λbullock(angmom::Vector{<:T}, mass::T, rad::T) where T <: Real
    # [G] = (Msun / h)^-1 (Mpc / h) (km / s)^2
    G = T(4.300917270069976e-09)
    λ = sqrt(sum(angmom.^2.0)) / sqrt(2 * G * mass^3.0 * rad)
    return T(λ)
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


"""
    nfw_concentration(halo::Halo{T}, rad::T, npart_min::Int=10) where T <: Real

Calculate the NFW concentration parameter for a given `halo` and maximum radius `rad`.

# Arguments
- `halo`: Halo object sorted by distance from its center of mass.
- `rad`: Radius for the concentration calculation.
- `npart_min` (optional): Minimum number of particles. Default is 10.

# Returns
- The computed NFW concentration parameter or NaN if the calculation fails.
"""
function nfw_concentration(halo::Halo{T}, rad::T, npart_min::Int=10) where T <: Real
    @assert halo.is_sorted "Halo must be sorted by distance from its CM"

    imax = find_first_above_threshold(halo.dist, rad)
    if imax === nothing || imax < npart_min
        return T(NaN)
    end

    imax -= 1

    log_conc_min = T(-3)
    log_conc_max = T(3)

    res = optimize(
        x -> negll_nfw_concentration(x, halo.dist / rad, halo.mass / halo.mass[1], imax),
        T(-3), T(3))

    if !Optim.converged(res)
        return T(NaN)
    end

    res = Optim.minimizer(res)

    if isapprox(res, log_conc_min) || isapprox(res, log_conc_max)
        return T(NaN)
    end

    return 10^res
end


function negll_nfw_concentration(log_c::T, xs::Vector{T}, ws::Vector{T}, imax::Int) where T <: Real
    c = 10^log_c

    negll = T(0)

    @turbo for i in 1:imax
        negll += log(ws[i] * xs[i] / (1 + c * xs[i])^2)
    end

    negll += @fastmath imax * log(c^2 / (log(1 + c)  - c / (1 + c)))
    negll *= -1

    return negll
end


"""
    inertia_tensor(halo, rad)

Compute the inertia tensor for a `halo` within a given radius `rad`.

# Arguments
- `halo`: Halo object sorted by distance from its center of mass.
- `rad`: Radius within which to compute the inertia tensor.

# Returns
- A 3x3 matrix representing the inertia tensor.
"""
function inertia_tensor(halo::Halo{T}, rad::T) where T <: Real
    @assert halo.is_sorted "Halo must be sorted by distance from its CM"

    imax = find_first_above_threshold(halo.dist, rad)
    if imax === nothing
        return NaN
    end

    imax -= 1

    M = sum(halo.mass[i] for i in 1:imax)

    pos = copy(halo.pos)
    shift_pos_to_center_of_box!(pos, halo.cm, halo.boxsize)

    Iij = zeros(T, 3, 3)
    @inbounds @fastmath for i in 1:3
        @inbounds @fastmath for j in 1:3
            if i > j
                Iij[i, j] = Iij[j, i]
                continue
            end

            @inbounds @fastmath for n in 1:imax
                Iij[i, j] += halo.mass[n] * pos[n, i] * pos[n, j]
            end

            Iij[i, j] /= M
        end
    end

    return  Iij
end


function sqrt_nan(x::T) where T <: Real
    if x < 0
        return T(NaN)
    end

    return sqrt(x)
end


"""
    ellipsoid_axes_ratio(Iij)

Calculate the axes ratios of an ellipsoid defined by its inertia tensor `Iij`.

# Arguments
- `Iij`: 3x3 inertia tensor matrix.

# Returns
- The axes ratios `(b/a, c/a)` for a > b > c.
"""
function ellipsoid_axes_ratio(Iij::Matrix{T}) where T <: Real
    @assert size(Iij) == (3, 3) "Iij must be a 3x3 matrix"
    c, b, a = sqrt_nan.(eigvals(Iij))
    q = b / a
    s = c / a

    isapprox(q, 0) ? q = T(NaN) : nothing
    isapprox(s, 0) ? s = T(NaN) : nothing

    return q, s
end


"""
    cm_displacement(halo)

Compute the displacement between the center of mass of a `halo` from the shrinking sphere
calculation and the total center of mass.

# Arguments
- `halo`: Halo object sorted by distance from its center of mass.

# Returns
-  The displacement between the two "center" of mass positions.
"""
function cm_displacement(halo::Halo{T}) where T <: Real
    @assert halo.is_sorted "Halo must be sorted by distance from its CM"

    cmtot = fill(T(NaN), 3)

    center_of_mass!(cmtot, halo.pos, halo.mass, halo.boxsize)

    return sqrt(sum((cmtot[i] - halo.cm[i])^2 for i in 1:3))
end


"""
    virial_fraction(halo, rad)

Compute the virial fraction of a `halo` within a given radius `rad`. Note that the halo must
have the potential of particles.

# Arguments
- `halo`: Halo object sorted by distance from its center of mass.

# Returns
- The virial fraction.
"""
function virial_fraction(halo::Halo{T}, rad::T) where T <: Real
    @assert halo.is_sorted "Halo must be sorted by distance from its CM"

    imax = find_first_above_threshold(halo.dist, rad)
    kinetic_energy = T(0.)
    potential_energy = T(0.)


    @inbounds @fastmath for i in 1:imax
        particle_kinetic_energy = T(0)
        @inbounds @fastmath for j in 1:3
            particle_kinetic_energy += halo.vel[i, j]^2
        end
        particle_kinetic_energy *= halo.mass[i] / 2

        kinetic_energy += particle_kinetic_energy
        potential_energy += halo.mass[i] * abs(halo.potential[i])
    end

    return 2 * kinetic_energy / potential_energy

end


