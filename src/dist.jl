"""
    periodic_distance(points::Array{T, 2}, reference::Vector{T}, boxsize::T) where T <: Real

Calculate the distance between a set of points and a reference point within a periodic box.

# Parameters
- `points`: A `Nx3` array of particle positions.
- `mass`: A `N` long vetor of particle masses.
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
    center_of_mass(points::Array{T, 2}, mass::Vector{T}, boxsize::T) where T <: Real

Calculate the center of mass of particles within a periodic box.

# Parameters
- `points`: A `Nx3` array of particle positions.
- `mass`: A `N` long vetor of particle masses.
- `boxsize`: Size of the box.

# Keyword Arguments
- `mask`: A `N` long vetor indicating which particles to include in the calculation.

# Returns
- A 1D array of size `3` indicating the center of mass coordinates `(x, y, z)` within the box.
"""
function center_of_mass(points::Array{T, 2}, mass::Vector{T}, boxsize::T;
                        mask::Union{Nothing, Vector{Bool}}=nothing) where T <: Real
    cm = zeros(T, 3)

    # Convert positions to unit circle coordinates in the complex plane,
    # calculate the weighted average and convert it back to box coordinates.
    for i in 1:3
        cm_i = 0.
        for j in 1:size(points, 1)
            if mask === nothing || mask[j]
                cm_i += mass[j] * exp.(2im * π * points[j, i] / boxsize)
            end
        end

        cm_i = atan(imag(cm_i), real(cm_i)) * boxsize / (2 * π)
        cm_i = cm_i < 0 ? cm_i + boxsize : cm_i

        cm[i] = cm_i
    end

    return cm
end


"""
    shrinking_sphere_cm(points::Array{T, 2}, mass::Vector{T}, boxsize::T;
                        npart_min=30, shrink_factor=0.98) where T <: Real

Compute the center of mass (CM) using a shrinking sphere approach.

# Arguments
- `points`: A `Nx3` array of particle positions.
- `mass`: A `N` long vetor of particle masses.
- `boxsize`: Size of the box.

# Keyword Arguments
- `npart_min`: Minimum number of particles. Once the number of particles within the current sphere radius
  falls below this threshold, the function will return the current CM.
- `shrink_factor`: Factor by which the sphere's radius is multiplied in each iteration.

# Returns
- A tuple containing the computed center of mass and the distances of points from the center of mass.
"""
function shrinking_sphere_cm(points::Array{T, 2}, mass::Vector{T}, boxsize::T;
                             npart_min=30, shrink_factor=0.98) where T <: Real

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
