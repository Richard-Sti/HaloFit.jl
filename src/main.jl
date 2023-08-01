"""
    periodic_distance(points::Array{T, 2}, reference::Vector{T}, boxsize::T) where T <: Real

Calculate the distance between a set of points and a reference point within a periodic box.

# Parameters
- `points`: A `Nx3` array of particle positions.
- `reference`: Reference point position.
- `boxsize`: Size of the box.

# Returns
- A 1D vector of length `N` containing the periodic distances between each point in `points` and the `reference` point.
"""
function periodic_distance(points::Array{T, 2}, reference::Vector{T}, boxsize::T) where T <: Real
    npoints = size(points, 1)
    dist = zeros(T, npoints)

    periodic_distance!(dist, points, reference, boxsize)

    return dist
end


function periodic_distance!(dist::Vector{T}, points::Array{T, 2}, reference::Vector{T}, boxsize::T) where T <: Real
    halfbox = boxsize / 2.0

    for i in 1:size(points, 1)
        dist_sq = 0.0
        for j in 1:3
            dist_1d = abs(points[i, j] - reference[j])
            dist_1d = dist_1d > halfbox ? boxsize - dist_1d : dist_1d
            dist_sq += dist_1d^2
        end

        dist[i] = sqrt(dist_sq)
    end

    return dist
end


"""
    function center_of_mass(points::Array{T, 2}, mass::Union{Vector{T}, Real}, boxsize::T;
                            mask::Union{Nothing, Vector{Bool}}=nothing) where T <: Real

Calculate the center of mass of particles within a periodic box.

# Parameters
- `points`: A `Nx3` array of particle positions.
- `mass`: Either a single mass value or a vector of masses corresponding to each distance in `dist`.
- `boxsize`: Size of the box.

# Keyword Arguments
- `mask`: A `N` long vetor indicating which particles to include in the calculation.

# Returns
- A 1D array of size `3` indicating the center of mass coordinates `(x, y, z)` within the box.
"""
function center_of_mass(points::Array{T, 2}, mass::Union{Vector{T}, Real}, boxsize::T;
                        mask::Union{Nothing, Vector{Bool}}=nothing) where T <: Real
    cm = zeros(T, 3)

    for i in 1:3
        cm_i = 0.
        for j in 1:size(points, 1)
            _mass = typeof(mass) <: Real ? mass : mass[j]
            if mask === nothing || mask[j]
                cm_i += _mass * exp.(2im * π * points[j, i] / boxsize)
            end
        end

        cm_i = atan(imag(cm_i), real(cm_i)) * boxsize / (2 * π)
        cm_i = cm_i < 0 ? cm_i + boxsize : cm_i

        cm[i] = cm_i
    end

    return cm
end


"""
    function shrinking_sphere_cm(points::Array{T, 2}, mass::Union{Vector{T}, Real}, boxsize::T;
                                 npart_min::Int=30, shrink_factor::Real=0.98) where T <: Real

Compute the center of mass (CM) using a shrinking sphere approach.

# Arguments
- `points`: A `Nx3` array of particle positions.
- `mass`: Either a single mass value or a vector of masses corresponding to each distance in `dist`.
- `boxsize`: Size of the box.

# Keyword Arguments
- `npart_min`: Minimum number of particles. Once the number of particles within the current sphere radius
  falls below this threshold, the function will return the current CM.
- `shrink_factor`: Factor by which the sphere's radius is multiplied in each iteration.

# Returns
- A tuple containing the computed center of mass and the distances of points from the center of mass.
"""
function shrinking_sphere_cm(points::Array{T, 2}, mass::Union{Vector{T}, Real}, boxsize::T;
                             npart_min::Int=30, shrink_factor::Real=0.98) where T <: Real

    npoints = size(points, 1)

    cm = center_of_mass(points, mass, boxsize)
    dist = zeros(T, npoints)
    within_rad = fill(true, npoints)

    rad = nothing

    while true
        dist = periodic_distance!(dist, points, cm, boxsize)

        if rad === nothing
            rad = maximum(dist)
        end

        for i in 1:npoints
            within_rad[i] = dist[i] <= rad
        end

        cm = center_of_mass(points, mass, boxsize; mask=within_rad)

        if sum(within_rad) < npart_min
            return cm, periodic_distance!(dist, points, cm, boxsize)
        end

        rad *= shrink_factor

    end
end


"""
    spherical_overdensity_mass!(dist::Vector{T}, mass::Union{Vector{T}, Real}, ρ_target::Real) where T <: Real

Calculate the spherical overdensity mass and radius around a CM, defined as the inner-most
radius where the density falls below a given threshold. The exact radius is found via linear
interpolation between the two particles enclosing the threshold.

# Arguments
- `dist`: Distance of each particle from the centre of mass.
- `mass`: Either a single mass value or a vector of masses corresponding to each distance in `dist`.
- `ρ_target`: The target density threshold.

# Returns
- `mass_in_rad`: Overdensity mass up to the radius where the density falls below `ρ_target` in (Msun / h).
- `rad`: Overdensity radius in box units where the density falls below `ρ_target`.

# Note
The function modifies the input `dist` in-place when `mass` is of type `Real`. Make sure that units of
`dist`, `mass` and `ρ_target` are consistent. The code does not perform any unit conversions and does not
assume the particles to be sorted.
"""
function spherical_overdensity_mass(dist::Vector{T}, mass::Union{Vector{T}, Real}, ρ_target::Real) where T <: Real
    copy(dist)

    if typeof(mass) <: Real
        sort!(dist)
        ρ = typeof(mass).(1:length(dist)) * mass
    else
        argsort = sortperm(dist)
        dist = dist[argsort]
        ρ = cumsum(mass[argsort])
    end

    totmass = ρ[end]
    ρ ./= (4 / 3 * π * dist.^3)
    ρ ./= ρ_target

    j = find_first_below_threshold(ρ, 1.)

    if j === nothing
        return NaN, NaN
    end

    i = j - 1

    rad = (dist[j] - dist[i]) * (1. - ρ[i]) / (ρ[j] - ρ[i]) + dist[i]
    mass_in_rad = (4 / 3 *  π * ρ_target) .* rad.^3

    if mass_in_rad > totmass
        return NaN, NaN
    end

    return mass_in_rad, rad
end


function find_first_below_threshold(x::Vector{<:Real}, threshold::Real)
    for i in 2:length(x)
        if x[i] < threshold
            return i
        end
    end

    return nothing
end


"""
    angular_momentum(pos::Array{T, 2}, vel::Array{T, 2}, mass::Union{Vector{T}, T}, cm::Vector{T}, boxsize::T) where T <: Real

Calculate the angular momentum around a given center of mass using particle properties.

# Arguments
- `pos`: An `(n_points, 3)` array representing the positions of each particle in space.
- `vel`: An `(n_points, 3)` array representing the velocities of each particle.
- `mass`: A vector of length `n_points` containing the masses of each particle or a single scalar value representing a uniform mass for all particles.
- `cm`: A vector of length 3 representing the center of mass in terms of x, y, and z coordinates.
- `boxsize`: The size of the box.

# Returns
- `angmom`: A vector of length 3 representing the angular momentum components (Lx, Ly, Lz) of the system around the specified center of mass.

# Notes
- The function accounts for the periodicity of the box when computing positions relative to the center of mass.
- The units of the returned angular momentum are simply the product of the units of the input positions, velocities, and masses.
"""
function angular_momentum(pos::Array{T, 2}, vel::Array{T, 2}, mass::Union{Vector{T}, T}, cm::Vector{T}, boxsize::T) where T <: Real
    pos = copy(pos)
    vel = copy(vel)


    shift_pos_to_center_of_box!(pos, cm, boxsize, true)
    shift_vel_to_cm_frame!(vel, mass)

    angmom = zeros(T, 3)
    npoints = size(pos, 1)
    for i in 1:npoints
        _mass = typeof(mass) <: Real ? mass : mass[i]
        angmom[1] += _mass * (pos[i, 2] * vel[i, 3] - pos[i, 3] * vel[i, 2])
        angmom[2] += _mass * (pos[i, 3] * vel[i, 1] - pos[i, 1] * vel[i, 3])
        angmom[3] += _mass * (pos[i, 1] * vel[i, 2] - pos[i, 2] * vel[i, 1])
    end

    return angmom
end


"""
    function lambda_bullock(angmom::Vector{T}, mass::T, rad::T) where T <: Real

Calculate the Bullock spin, see Eq. 5 in [1].

# Parameters
- `angmom`: Angular momentum in (Msun / h) * (Mpc / h) * (km / s).
- `mass`: Mass in Msun / h.
- `rad`: Radius corresponding to `mass` in Mpc / h.

# Returns
- `lambda_bullock`: float

# References
[1] A Universal Angular Momentum Profile for Galactic Halos; 2001;
Bullock, J. S.; Dekel, A.;  Kolatt, T. S.; Kravtsov, A. V.;
Klypin, A. A.; Porciani, C.; Primack, J. R.

# Notes
- The input quantities should only be calculated with particles in some radius.
"""
function lambda_bullock(angmom::Vector{T}, mass::T, rad::T) where T <: Real
    G = 4.300917270069976e-09  # G in (Msun / h)^-1 (Mpc / h) (km / s)^2
    return sqrt(sum(angmom.^2)) / sqrt(2 * G * mass^3 * rad)
end


function shift_pos_to_center_of_box!(points::Array{T, 2}, cm::Vector{T}, boxsize::T,
                                     set_cm_to_zero::Bool=false) where T <: Real
    halfboxsize = boxsize / 2

    for i in 1:size(points, 1)
        for j in 1:3
            points[i, j] += (halfboxsize - cm[j])
            points[i, j] %= boxsize

            if set_cm_to_zero
                points[i, j] -= halfboxsize
            end

        end
    end
end


function shift_vel_to_cm_frame!(vel::Array{T, 2}, mass::Union{Vector{T}, Real}) where T <: Real
    npoints = size(vel, 1)
    totmass = typeof(mass) <: Real ? npoints * mass : sum(mass)
    for i in 1:3
        vel_i = 0.

        for j in 1:npoints
            m = typeof(mass) <: Real ? mass : mass[j]
            vel_i += m * vel[j, i]
        end

        vel_i /= totmass
        vel[:, i] .-= vel_i

    end
    return vel
end


"""
    crit_density0(h)

Return the z = 0 critical density in units of h^2 Msun / Mpc^3.
"""
function crit_density0(h::Real)
    return 2.77536627e+11 * h^2
end
