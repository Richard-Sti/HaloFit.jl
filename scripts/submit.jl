using ArgParse

include("fit_script.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table! s begin
        "--kind"
        help = "Kind of simulation"
    end

    return parse_args(s)
end

args = parse_commandline()

if args["kind"] == "csiborg"
    fit_csiborg()
elseif args["kind"] == "tng300dark"
    fit_tng300dark()
else
    error("Unknown simulation kind: `$(args["kind"])`")
end
