using ProgressMeter

using Pkg
Pkg.activate("/mnt/zfsusers/rstiskalek/HaloFit")

using HaloFit
using NPZ
using HDF5
using DataFrames


function zfill(n::String, width::Int)
    return lpad(n, width, '0')
end


function zfill(n::Int, width::Int)
    return zfill(string(n), width)
end



################################################################################
#                        I/O of the particles                                  #
################################################################################

function make_offsets(halomap)
    if isa(halomap, HDF5.Dataset)
        halomap = halomap[:, :]
    end

    return Dict(halomap[1, i] => Int64.(halomap[2:end, i]) for i in 2:size(halomap, 2));
end


function load_halo_from_offsets(hid, particles, offsets, boxsize)
    i, j = offsets[hid]
    pos = Matrix(particles[1:3, i:j]')
    vel = Matrix(particles[4:6, i:j]')
    mass = particles[7, i:j]

    pos .*= eltype(pos).(boxsize)
    return pos, vel, mass
end


function save_frame(fout, df)
    println("Writing to ... `$(fout)`")
    h5open(fout, "w") do file
        for col in names(df)
            file[col] = df[!, col]
        end
    end
end


################################################################################
#                             CSiBORG Fitting                                  #
################################################################################


function fit_single_csiborg(fpath::String, load_in_memory::Bool=false)
    boxsize = Float32(677.7)
    f = h5open(fpath, "r")
    offsets = make_offsets(f["halomap"])


    ρ200c = crit_density0(1.) * 200

    symbols = [:cmx, :cmy, :cmz, :mtot, :m200c, :r200c, :lambda200c]
    n_cols = length(symbols)
    # n_rows = length(offsets)
    n_rows = 5000

    df = DataFrame([fill(NaN, n_rows) for _ in 1:n_cols], symbols)

    p = Progress(n_rows; showspeed=true)

    particles = f["particles"]

    if load_in_memory
        particles = particles[:, :]
    end

    for i in 1:n_rows
        pos, vel, mass = load_halo_from_offsets(i, particles, offsets, boxsize)
        cm, dist = shrinking_sphere_cm(pos, mass, boxsize)

        m200c, r200c = spherical_overdensity_mass(dist, mass, ρ200c)

        mask = dist .< r200c
        angmom = angular_momentum(pos[mask, :], vel[mask, :], mass[mask], cm, boxsize)
        λ200c = lambda_bullock(angmom, m200c, r200c)

        df[i, :cmx] = cm[1]
        df[i, :cmy] = cm[2]
        df[i, :cmz] = cm[3]
        df[i, :mtot] = sum(mass)
        df[i, :m200c] = m200c
        df[i, :r200c] = r200c
        df[i, :lambda200c] = λ200c

        next!(p)
    end

    return df

end


################################################################################
#                             Submission                                       #
################################################################################


particles_path(nsim::Integer) = "/mnt/extraspace/rstiskalek/CSiBORG/particles/parts_$(zfill(nsim, 5)).h5"
output_path(nsim::Integer) = "/mnt/zfsusers/rstiskalek/HaloFit//scripts/fit_$(zfill(nsim, 5)).h5"


nsims = [7444]

for nsim in nsims
    println("Calculating for `$(nsim)`")
    df = fit_single_csiborg(particles_path(nsim))
    save_frame(output_path(nsim), df)
end
