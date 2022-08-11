### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 4d5ceb64-18e2-40b6-b6ab-9a7befbe27b2
begin
	using LinearAlgebra: LinearAlgebra, dot
	using Test: @test
	using Random: Random
end

# ╔═╡ 85316f5e-12ad-4aca-b1d4-9fc2a66d5469
rng = Random.MersenneTwister(1234)

# ╔═╡ 1d0075aa-5c04-4466-9325-37bb133cd37b


# ╔═╡ 2368d0f8-2316-4b2e-809d-7eac85d4633f


# ╔═╡ e232fd87-92eb-4f82-8374-373d9b3d317c
struct PackedLowerTriangular{T}
	data::Vector{T}
end


# ╔═╡ 9b629aba-73e8-49ba-b31f-03e89b430146


# ╔═╡ de992a3c-c568-46a6-9555-48838ad7045e
# eventually want to call LAPACK https://netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2_gae6fb0355e398779dc593ced105ce373d.html#gae6fb0355e398779dc593ced105ce373d
function \(A::PackedLowerTriangular{T}, v::Vector{T}) where T
	n = length(v)
	result = Vector{T}(undef, n)
	p = 0
	for idx in 0:(n-1)
		result[idx+1] = 
			(v[idx+1] - dot(result[1:idx], A.data[(p+1):p+idx]))/A.data[p+idx+1]
		p += idx + 1
	end
	return result
end

# ╔═╡ b413c950-197f-11ed-2b4b-73333a1275ac
begin
	mutable struct GaussianRandomField{T}
		cov::Function
		randomness::Vector{T}
		evaluated_points::Array{T, 2}
		chol_cov::PackedLowerTriangular{T}
		evaluations::Vector{T}

		GaussianRandomField{T}(cov::Function) where T = new(
			cov
		)
	end

	"""
	evaluate random field at point x
	"""
	function (rf::GaussianRandomField{T})(x::Vector{T}) where {T}
		try
			cross_cov::Vector{T} = map(eachcol(rf.evaluated_points)) do pt
				rf.cov(pt, x)
			end
			rf.evaluated_points = [rf.evaluated_points x]
			coeff = rf.chol_cov \ cross_cov
			cond_σ = sqrt(rf.cov(x,x) - dot(coeff, coeff))
			push!(rf.chol_cov.data, coeff..., cond_σ)
			push!(rf.randomness, randn(rng, T))
			push!(
				rf.evaluations, 
				dot(coeff, rf.randomness[1:end-1]) + cond_σ * rf.randomness[end]
			)
		catch e
			e isa UndefRefError || throw(e)
			# unevaluated random field
			rf.evaluated_points = reshape(x, :, 1)
			rf.randomness = [randn(rng, T)]
			σ = sqrt(rf.cov(x,x))
			rf.chol_cov = PackedLowerTriangular{T}([σ])
			rf.evaluations = [σ * rf.randomness[end]]
		end
		return rf.evaluations[end]
	end
end

# ╔═╡ beaf95e8-10f0-4b34-be36-2ba7825a7d17
function squaredExponentialKernel(x,y)
	return exp(-LinearAlgebra.norm(x-y)^2/2)
end

# ╔═╡ 51be2a30-538d-4d10-bb69-53c0aac3d92f
rf = GaussianRandomField{Float64}(squaredExponentialKernel)

# ╔═╡ fe4bee1e-f0a3-4b2c-8be7-301647114d1a
begin
	rf([0.,])
	rf([1.])
	rf.evaluated_points
	rf.evaluations
end

# ╔═╡ e83a078d-02b3-4c9f-a8cf-2bdeed5dbff0
a = [1 2 ;3 4]

# ╔═╡ 4f3b842c-e631-47c0-aa6c-f8a145be523a
reshape(a, :,1)

# ╔═╡ 4a88596a-0bb9-4a36-a663-aff609290f1f
md"# Tests"

# ╔═╡ 99455ed7-7c20-4162-8967-c88c4bbabe23
begin
	packed = PackedLowerTriangular([1,1,1])
	@test packed\[2,2] == [2,0]
	@test packed\[1,2] == [1,1]
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.7.3"
manifest_format = "2.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl", "OpenBLAS_jll"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
"""

# ╔═╡ Cell order:
# ╠═4d5ceb64-18e2-40b6-b6ab-9a7befbe27b2
# ╠═85316f5e-12ad-4aca-b1d4-9fc2a66d5469
# ╠═1d0075aa-5c04-4466-9325-37bb133cd37b
# ╠═2368d0f8-2316-4b2e-809d-7eac85d4633f
# ╠═e232fd87-92eb-4f82-8374-373d9b3d317c
# ╠═9b629aba-73e8-49ba-b31f-03e89b430146
# ╠═de992a3c-c568-46a6-9555-48838ad7045e
# ╠═b413c950-197f-11ed-2b4b-73333a1275ac
# ╠═beaf95e8-10f0-4b34-be36-2ba7825a7d17
# ╠═51be2a30-538d-4d10-bb69-53c0aac3d92f
# ╠═fe4bee1e-f0a3-4b2c-8be7-301647114d1a
# ╠═e83a078d-02b3-4c9f-a8cf-2bdeed5dbff0
# ╠═4f3b842c-e631-47c0-aa6c-f8a145be523a
# ╟─4a88596a-0bb9-4a36-a663-aff609290f1f
# ╟─99455ed7-7c20-4162-8967-c88c4bbabe23
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
