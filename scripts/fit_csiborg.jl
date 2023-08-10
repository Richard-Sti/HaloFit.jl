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
using Pkg
Pkg.activate("/mnt/zfsusers/rstiskalek/HaloFit")

using DataFrames, HaloFit, HDF5, ProgressMeter


begin
    zfill(n::Int, width::Int) = zfill(string(n), width)
    zfill(n::String, width::Int) = lpad(n, width, '0')
    path_csiborg_particles(nsim::Integer) = "/mnt/extraspace/rstiskalek/CSiBORG/particles/parts_$(zfill(nsim, 5)).h5"
    path_csiborg_output(nsim::Integer) = "/mnt/extraspace/rstiskalek/CSiBORG/structfit/halos_$(zfill(nsim, 5)).hdf5"
    path_tng300dark_particles() = "/mnt/extraspace/rstiskalek/TNG300-1-Dark/sorted_halos.hdf5"
    path_tng300dark_output() = "/mnt/extraspace/rstiskalek/TNG300-1-Dark/fitted_halos.hdf5"
end


function make_offsets(halomap)
    if isa(halomap, HDF5.Dataset)
        halomap = halomap[:, :]
    end

    return Dict(halomap[1, i] => Int64.(halomap[2:end, i]) for i in 2:size(halomap, 2));
end


function load_halo_from_offsets(hid, particles, offsets, boxsize, simname; mpart=nothing)
    i, j = offsets[hid]
    i += 1  # Because Python has different indexing..
    pos = Matrix(particles[1:3, i:j]')
    vel = Matrix(particles[4:6, i:j]')

    if isnothing(mpart)
        mass = particles[7, i:j]
    else
        mass = fill(eltype(pos)(mpart), j - i + 1)
    end

    if simname == "csiborg"
        pos .*= @fastmath eltype(pos).(boxsize)
    elseif simname == "tng300dark"
        pos ./= @fastmath 1000
    else
        error("Unknown simulation name: `$(simname)`")
    end

    return Halo(pos, vel, mass, boxsize)
end


function fit_from_offsets(fpath, boxsize, simname;
                          verbose::Bool=true, load_in_memory::Bool=false, mpart=nothing,
                          npart_min=50, shrink_factor=0.975)
    f = h5open(fpath, "r")
    particles = f["particles"]

    if simname == "csiborg"
        offsets = make_offsets(f["halomap"])
    elseif simname == "tng300dark"
        offsets = make_offsets(f["offsets"])
    else
        error("Unknown simulation name: `$(simname)`")
    end

    ρ200c = Float32(ρcrit0(1) * 200)
    symbols = [:hid, :cmx, :cmy, :cmz, :mtot, :m200c, :r200c, :lambda200c, :conc, :q, :s]
    n_cols = length(symbols)
    n_rows = length(offsets)

    if load_in_memory
        println("Loading particles into memory ...")
        t0 = time()
        particles = particles[:, :]
        println("Loaded particles in in $(time() - t0) seconds.")
    end

    df = DataFrame([fill(Float32(NaN), n_rows) for _ in 1:n_cols], symbols)
    p = Progress(n_rows; enabled=verbose, dt=0.1, barlen=50, showspeed=true)
    for i in 1:n_rows
        halo = load_halo_from_offsets(i, particles, offsets, boxsize, simname; mpart=mpart)
        df[i, :hid] = offsets[i][1]

        if length(halo) < 100
            next!(p)
            continue
        end

        df[i, :mtot] = sum(halo.mass)

        shrinking_sphere_cm!(halo; npart_min=npart_min, shrink_factor=shrink_factor)
        df[i, :cmx] = halo.cm[1]
        df[i, :cmy] = halo.cm[2]
        df[i, :cmz] = halo.cm[3]

        m200c, r200c = spherical_overdensity_mass(halo, ρ200c)
        df[i, :m200c] = m200c
        df[i, :r200c] = r200c

        if !isnan(m200c)
            angmom = angular_momentum(halo, r200c)
            df[i, :lambda200c] = λbullock(angmom, m200c, r200c)

            df[i, :conc] = nfw_concentration(halo, r200c)

            Iij = inertia_tensor(halo, r200c)
            q, s = ellipsoid_axes_ratio(Iij)
            df[i, :q] = q
            df[i, :s] = s
        end

        next!(p)
    end
    finish!(p)

    return df
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
#                               Submission                                     #
################################################################################


function fit_csiborg()
    nsims = [7444 + n * 24 for n in 0:100]
    boxsize = 677.7  # Mpc/h
    for nsim in nsims
        println("Fitting CSiBORG IC `$(nsim)`")

        fpath = path_csiborg_particles(nsim)
        res = fit_from_offsets(fpath, boxsize, "csiborg"; verbose=true);

        fout = path_csiborg_output(nsim)
        println("Saving to ... `$(fout)`")
        save_frame(fout, res)
    end
end


function fit_tng300dark()
    mpart = 0.0047271638660809 * 1e10   # Msun/h
    boxsize = 205.0                     # Mpc/h

    fpath = path_tng300dark_particles()
    res = fit_from_offsets(fpath, boxsize, "tng300dark"; verbose=true, mpart=mpart,
                           npart_min=250, shrink_factor=0.925)

    fout = path_tng300dark_output()
    println("Saving to ... `$(fout)`")
    save_frame(fout, res)
end