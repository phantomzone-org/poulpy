In this folder, we deal with the full complex FFT in `C[X] mod X^M-i`.
One complex is represented by two consecutive doubles `(real,imag)`
Note that a real polynomial sum_{j=0}^{N-1} p_j.X^j mod X^N+1 
corresponds to the complex polynomial of half degree `M=N/2`: 
`sum_{j=0}^{M-1} (p_{j} + i.p_{j+M}) X^j mod X^M-i`

For a complex polynomial A(X) sum c_i X^i of degree M-1
or a real polynomial sum a_i X^i of degree N

coefficient space: 
a_0,a_M,a_1,a_{M+1},...,a_{M-1},a_{2M-1}
or equivalently
Re(c_0),Im(c_0),Re(c_1),Im(c_1),...Re(c_{M-1}),Im(c_{M-1})

eval space:
c(omega_{0}),...,c(omega_{M-1})

where
omega_j = omega^{1+rev_{2N}(j)}
and omega = exp(i.pi/N)

rev_{2N}(j) is the number that has the log2(2N) bits of j in reverse order.