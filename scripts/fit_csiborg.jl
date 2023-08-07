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
    path_csiborg_output(nsim::Integer) = "/mnt/zfsusers/rstiskalek/HaloFit/scripts/fit_$(zfill(nsim, 5)).h5"
end


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

    pos .*= @fastmath eltype(pos).(boxsize)

    return Halo(pos, vel, mass, boxsize)
end


function fit_from_offsets(fpath, boxsize; verbose::Bool=true, load_in_memory::Bool=false)
    f = h5open(fpath, "r")
    particles = f["particles"]
    offsets = make_offsets(f["halomap"])

    ρ200c = Float32(ρcrit0(1) * 200)
    symbols = [:cmx, :cmy, :cmz, :mtot, :m200c, :r200c, :lambda200c, :conc, :q, :s]
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
        halo = load_halo_from_offsets(i, particles, offsets, boxsize)

        if length(halo) < 100
            next!(p)
            continue
        end

        df[i, :mtot] = sum(halo.mass)

        shrinking_sphere_cm!(halo)
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
    nsims = [7444, 7444 + 24, 7444 + 24 * 2]
    for nsim in nsims
        println("Fitting CSiBORG IC `$(nsim)`")
        fpath = path_csiborg_particles(nsim)
        res = fit_from_offsets(fpath, 677.7; verbose=true, load_in_memory=false);

        # TODO: save to HDF5
    end
end
