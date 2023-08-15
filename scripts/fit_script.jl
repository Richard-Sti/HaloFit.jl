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
using DataFrames, HaloFit, HDF5, ProgressMeter


begin
    zfill(n::Int, width::Int) = zfill(string(n), width)
    zfill(n::String, width::Int) = lpad(n, width, '0')
    path_csiborg_particles(nsim::Integer) = "/mnt/extraspace/rstiskalek/CSiBORG/particles/parts_$(zfill(nsim, 5)).h5"
    path_tng300dark_particles() = "/mnt/extraspace/rstiskalek/TNG300-1-Dark/sorted_halos.hdf5"
end


function make_offsets(halomap::HDF5.Dataset;
                      start_index::Integer,
                      zero_index::Bool)
    x = halomap[:, :]
    hids = x[1, start_index:end]

    offsets = Dict{Int64, Vector{Int64}}()
    for n in start_index:size(x, 2)
        hid = x[1, n]
        i, j = x[2, n], x[3, n]

        if zero_index
            i += 1
            j += 1
        end

        offsets[hid] = [i, j]
    end

    return hids, offsets
end


function load_halo_from_offsets(hid::Integer, particles::HDF5.Dataset, offsets::Dict, boxsize::Real, simname::String;
                                mpart::Union{Nothing, Real}=nothing)
    i, j = offsets[hid]
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
        # TNG300-1-Dark is in kpc / h
        pos ./= @fastmath 1000
    else
        error("Unknown simulation name: `$(simname)`")
    end

    return Halo(pos, vel, mass, boxsize)
end


function fit_from_offsets(fpath::String, boxsize::Real, simname::String;
                          start_index::Integer,
                          zero_index::Bool,
                          verbose::Bool=true,
                          npart_min::Integer=100,
                          mpart::Union{Nothing, Real}=nothing,
                          shrink_npart_min::Int=50,
                          shrink_factor::Real=0.975)
    f = h5open(fpath, "r")
    particles = f["particles"]

    hids, offsets = make_offsets(f["halomap"]; start_index=2, zero_index=true)

    ρ200c = Float32(ρcrit0(1) * 200)
    symbols = [:hid, :cmx, :cmy, :cmz, :mtot, :m200c, :r200c, :lambda200c, :conc, :q, :s]
    n_cols = length(symbols)
    n_rows = length(offsets)

    df = DataFrame([fill(Float32(NaN), n_rows) for _ in 1:n_cols], symbols)
    p = Progress(n_rows; enabled=verbose, dt=1, barlen=50, showspeed=true)
    for i in 1:n_rows
        hid = hids[i]
        df[i, :hid] = hid

        halo = load_halo_from_offsets(hid, particles, offsets, boxsize, simname; mpart=mpart)

        if length(halo) < npart_min
            next!(p)
            continue
        end

        df[i, :mtot] = sum(halo.mass)

        shrinking_sphere_cm!(halo; npart_min=shrink_npart_min, shrink_factor=shrink_factor)
        df[i, :cmx], df[i, :cmy], df[i, :cmz] = halo.cm[1], halo.cm[2], halo.cm[3]


        m200c, r200c = spherical_overdensity_mass(halo, ρ200c)
        df[i, :m200c], df[i, :r200c] = m200c, r200c

        if !isnan(m200c)
            angmom = angular_momentum(halo, r200c)
            df[i, :lambda200c] = λbullock(angmom, m200c, r200c)

            df[i, :conc] = nfw_concentration(halo, r200c)

            Iij = inertia_tensor(halo, r200c)
            df[i, :q], df[i, :s] = ellipsoid_axes_ratio(Iij)
        end

        next!(p)
    end

    finish!(p)

    return df
end


function save_frame(fout::String, df::DataFrames.DataFrame)
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
    boxsize = 677.7  # Mpc/h
    for nsim in [7444 + n * 24 for n in 0:100]
        println("Fitting CSiBORG IC `$(nsim)`")

        res = fit_from_offsets(path_csiborg_particles(nsim), boxsize, "csiborg";
                               start_index=2, zero_index=true, npart_min=100,
                               verbose=true)

        fout = "/mnt/extraspace/rstiskalek/CSiBORG/structfit/halos_$(zfill(nsim, 5)).hdf5"
        save_frame(fout, res)

    end
end


function fit_tng300dark()
    mpart = 0.0047271638660809 * 1e10   # Msun/h
    boxsize = 205.0                     # Mpc/h

    println("Fitting TNG300-1-Dark")
    res = fit_from_offsets(path_tng300dark_particles(), boxsize, "tng300dark";
                           start_index=1, zero_index=true, npart_min=100,
                           verbose=true, mpart=mpart,
                           shrink_npart_min=250, shrink_factor=0.95)

    fout = "/mnt/extraspace/rstiskalek/TNG300-1-Dark/fitted_halos.hdf5"
    save_frame(fout, res)
end
