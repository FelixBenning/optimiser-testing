### A Pluto.jl notebook ###
# v0.19.11

using Markdown
using InteractiveUtils

# ╔═╡ 4d5ceb64-18e2-40b6-b6ab-9a7befbe27b2
begin
	using LinearAlgebra: dot
	using Test: @test
end

# ╔═╡ e232fd87-92eb-4f82-8374-373d9b3d317c
struct PackedUpperTriangular{T}
	data::Vector{T}
end


# ╔═╡ de992a3c-c568-46a6-9555-48838ad7045e
# eventually want to call LAPACK https://netlib.org/lapack/explore-html/d6/d30/group__single__blas__level2_gae6fb0355e398779dc593ced105ce373d.html#gae6fb0355e398779dc593ced105ce373d
function \(A::PackedUpperTriangular{T}, v::Vector{T}) where T
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

# ╔═╡ 72e4c649-5ef1-4f44-afe5-9365e239d97e


# ╔═╡ b413c950-197f-11ed-2b4b-73333a1275ac
begin
	struct GaussianRandomField{T}
		chol_cov::PackedUpperTriangular{T}
	end
	
	function (rf::GaussianRandomField)(x) # evaluate random field at point x
		rf
	end
end

# ╔═╡ 99ab5b58-a340-4ee1-aaa0-6d280cbdfa3d


# ╔═╡ d4b9fb89-c04e-46b2-851c-eb7a343a26ef
a = [1,2,3]

# ╔═╡ fc08c5e0-924f-4593-b571-30fa64e34c3f
LinearAlgebra.dot(a[1:0], a[1:0])

# ╔═╡ bad01da8-ce00-43ff-8e24-c958679389f2


# ╔═╡ 99455ed7-7c20-4162-8967-c88c4bbabe23
begin
	packed = PackedUpperTriangular([1,1,1])
	@test packed\[2,2] == [2,0]
	@test packed\[1,2] == [1,1]
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
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
# ╠═e232fd87-92eb-4f82-8374-373d9b3d317c
# ╠═de992a3c-c568-46a6-9555-48838ad7045e
# ╠═72e4c649-5ef1-4f44-afe5-9365e239d97e
# ╠═b413c950-197f-11ed-2b4b-73333a1275ac
# ╠═99ab5b58-a340-4ee1-aaa0-6d280cbdfa3d
# ╠═d4b9fb89-c04e-46b2-851c-eb7a343a26ef
# ╠═fc08c5e0-924f-4593-b571-30fa64e34c3f
# ╠═bad01da8-ce00-43ff-8e24-c958679389f2
# ╟─99455ed7-7c20-4162-8967-c88c4bbabe23
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
