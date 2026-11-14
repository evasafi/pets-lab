import tenseal as ts

ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192, coeff_mod_bit_sizes=[60, 40, 40, 60])
ctx.global_scale = 2**40

a = ts.ckks_vector(ctx, [1.0, 2.0, 3.0])
b = ts.ckks_vector(ctx, [10.0, 20.0, 30.0])

print("a + b (computed encrypted):", [round(x, 4) for x in (a + b).decrypt()])
print("a * b (computed encrypted):", [round(x, 4) for x in (a * b).decrypt()])
