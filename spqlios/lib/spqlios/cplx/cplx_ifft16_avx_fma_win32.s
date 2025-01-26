        .text
        .p2align 4
        .globl  cplx_ifft16_avx_fma
        .def    cplx_ifft16_avx_fma;    .scl    2;      .type   32;     .endef
cplx_ifft16_avx_fma:

  pushq %rdi
  pushq %rsi
  movq %rcx,%rdi
  movq %rdx,%rsi
  subq $0x100,%rsp
  movdqu %xmm6,(%rsp)
  movdqu %xmm7,0x10(%rsp)
  movdqu %xmm8,0x20(%rsp)
  movdqu %xmm9,0x30(%rsp)
  movdqu %xmm10,0x40(%rsp)
  movdqu %xmm11,0x50(%rsp)
  movdqu %xmm12,0x60(%rsp)
  movdqu %xmm13,0x70(%rsp)
  movdqu %xmm14,0x80(%rsp)
  movdqu %xmm15,0x90(%rsp)
  callq cplx_ifft16_avx_fma_amd64
  movdqu (%rsp),%xmm6
  movdqu 0x10(%rsp),%xmm7
  movdqu 0x20(%rsp),%xmm8
  movdqu 0x30(%rsp),%xmm9
  movdqu 0x40(%rsp),%xmm10
  movdqu 0x50(%rsp),%xmm11
  movdqu 0x60(%rsp),%xmm12
  movdqu 0x70(%rsp),%xmm13
  movdqu 0x80(%rsp),%xmm14
  movdqu 0x90(%rsp),%xmm15
  addq $0x100,%rsp
  popq %rsi
  popq %rdi
  retq

# shifted FFT over X^16-i
# 1st argument (rdi) contains 16 complexes
# 2nd argument (rsi) contains: 8 complexes
#     omega,alpha,beta,j.beta,gamma,j.gamma,k.gamma,kj.gamma
#     alpha = sqrt(omega), beta = sqrt(alpha), gamma = sqrt(beta)
#     j = sqrt(i), k=sqrt(j)

cplx_ifft16_avx_fma_amd64:
vmovupd (%rdi),%ymm8        # load data into registers %ymm8 -> %ymm15
vmovupd 0x20(%rdi),%ymm9
vmovupd 0x40(%rdi),%ymm10
vmovupd 0x60(%rdi),%ymm11
vmovupd 0x80(%rdi),%ymm12
vmovupd 0xa0(%rdi),%ymm13
vmovupd 0xc0(%rdi),%ymm14
vmovupd 0xe0(%rdi),%ymm15

.fourth_pass:
vmovupd 0(%rsi),%ymm0                 /* gamma   */
vmovupd 32(%rsi),%ymm2                /* delta   */
vshufpd $15, %ymm0, %ymm0, %ymm1      /* ymm1: gama.iiii */
vshufpd $15, %ymm2, %ymm2, %ymm3      /* ymm3: delta.iiii */
vshufpd $0,  %ymm0, %ymm0, %ymm0      /* ymm0: gama.rrrr */
vshufpd $0,  %ymm2, %ymm2, %ymm2      /* ymm2: delta.rrrr */
vperm2f128 $0x31,%ymm10,%ymm8,%ymm4   # ymm4 contains c1,c5
vperm2f128 $0x31,%ymm11,%ymm9,%ymm5   # ymm5 contains c3,c7
vperm2f128 $0x31,%ymm14,%ymm12,%ymm6  # ymm6 contains c9,c13
vperm2f128 $0x31,%ymm15,%ymm13,%ymm7  # ymm7 contains c11,c15
vperm2f128 $0x20,%ymm10,%ymm8,%ymm8   # ymm8 contains c0,c4
vperm2f128 $0x20,%ymm11,%ymm9,%ymm9   # ymm9 contains c2,c6
vperm2f128 $0x20,%ymm14,%ymm12,%ymm10 # ymm10 contains c8,c12
vperm2f128 $0x20,%ymm15,%ymm13,%ymm11 # ymm11 contains c10,c14
vsubpd %ymm4,%ymm8,%ymm12  # tw:  to mul by gamma
vsubpd %ymm5,%ymm9,%ymm13  # itw: to mul by i.gamma
vsubpd %ymm6,%ymm10,%ymm14 # tw:  to mul by delta
vsubpd %ymm7,%ymm11,%ymm15 # itw: to mul by i.delta
vaddpd %ymm4,%ymm8,%ymm8
vaddpd %ymm5,%ymm9,%ymm9
vaddpd %ymm6,%ymm10,%ymm10
vaddpd %ymm7,%ymm11,%ymm11
vshufpd $5, %ymm12, %ymm12, %ymm4
vshufpd $5, %ymm13, %ymm13, %ymm5
vshufpd $5, %ymm14, %ymm14, %ymm6
vshufpd $5, %ymm15, %ymm15, %ymm7
vmulpd %ymm4,%ymm1,%ymm4
vmulpd %ymm5,%ymm0,%ymm5
vmulpd %ymm6,%ymm3,%ymm6
vmulpd %ymm7,%ymm2,%ymm7
vfmaddsub231pd  %ymm12, %ymm0, %ymm4     # ymm4 = (ymm0 * ymm12) +/- ymm4
vfmsubadd231pd  %ymm13, %ymm1, %ymm5
vfmaddsub231pd  %ymm14, %ymm2, %ymm6
vfmsubadd231pd  %ymm15, %ymm3, %ymm7

vperm2f128 $0x20,%ymm6,%ymm10,%ymm12  # ymm4 contains c1,c5 -- x gamma
vperm2f128 $0x20,%ymm7,%ymm11,%ymm13  # ymm5 contains c3,c7 -- x igamma
vperm2f128 $0x31,%ymm6,%ymm10,%ymm14  # ymm6 contains c9,c13 -- x delta
vperm2f128 $0x31,%ymm7,%ymm11,%ymm15  # ymm7 contains c11,c15 -- x idelta
vperm2f128 $0x31,%ymm4,%ymm8,%ymm10   # ymm10 contains c8,c12
vperm2f128 $0x31,%ymm5,%ymm9,%ymm11   # ymm11 contains c10,c14
vperm2f128 $0x20,%ymm4,%ymm8,%ymm8    # ymm8 contains c0,c4
vperm2f128 $0x20,%ymm5,%ymm9,%ymm9    # ymm9 contains c2,c6


.third_pass:
vmovupd 64(%rsi),%xmm0                /* gamma   */
vmovupd 80(%rsi),%xmm2                /* delta   */
vinsertf128 $1, %xmm0, %ymm0, %ymm0
vinsertf128 $1, %xmm2, %ymm2, %ymm2
vshufpd $15, %ymm0, %ymm0, %ymm1      /* ymm1: gama.iiii */
vshufpd $15, %ymm2, %ymm2, %ymm3      /* ymm3: delta.iiii */
vshufpd $0,  %ymm0, %ymm0, %ymm0      /* ymm0: gama.rrrr */
vshufpd $0,  %ymm2, %ymm2, %ymm2      /* ymm2: delta.rrrr */
vsubpd %ymm9,%ymm8,%ymm4
vsubpd %ymm11,%ymm10,%ymm5
vsubpd %ymm13,%ymm12,%ymm6
vsubpd %ymm15,%ymm14,%ymm7
vaddpd %ymm9,%ymm8,%ymm8
vaddpd %ymm11,%ymm10,%ymm10
vaddpd %ymm13,%ymm12,%ymm12
vaddpd %ymm15,%ymm14,%ymm14
vshufpd $5, %ymm4, %ymm4, %ymm9
vshufpd $5, %ymm5, %ymm5, %ymm11
vshufpd $5, %ymm6, %ymm6, %ymm13
vshufpd $5, %ymm7, %ymm7, %ymm15
vmulpd %ymm9,%ymm1,%ymm9
vmulpd %ymm11,%ymm0,%ymm11
vmulpd %ymm13,%ymm3,%ymm13
vmulpd %ymm15,%ymm2,%ymm15
vfmaddsub231pd  %ymm4, %ymm0, %ymm9     # ymm9 = (ymm0 * ymm4) +/- ymm9
vfmsubadd231pd  %ymm5, %ymm1, %ymm11
vfmaddsub231pd  %ymm6, %ymm2, %ymm13
vfmsubadd231pd  %ymm7, %ymm3, %ymm15

.second_pass:
vmovupd 96(%rsi),%xmm0                /* omri   */
vinsertf128 $1, %xmm0, %ymm0, %ymm0   /* omriri */
vshufpd $15, %ymm0, %ymm0, %ymm1      /* ymm1: omiiii */
vshufpd $0,  %ymm0, %ymm0, %ymm0      /* ymm0: omrrrr */
vsubpd %ymm10,%ymm8,%ymm4
vsubpd %ymm11,%ymm9,%ymm5
vsubpd %ymm14,%ymm12,%ymm6
vsubpd %ymm15,%ymm13,%ymm7
vaddpd %ymm10,%ymm8,%ymm8
vaddpd %ymm11,%ymm9,%ymm9
vaddpd %ymm14,%ymm12,%ymm12
vaddpd %ymm15,%ymm13,%ymm13
vshufpd $5, %ymm4, %ymm4, %ymm10
vshufpd $5, %ymm5, %ymm5, %ymm11
vshufpd $5, %ymm6, %ymm6, %ymm14
vshufpd $5, %ymm7, %ymm7, %ymm15
vmulpd %ymm10,%ymm1,%ymm10
vmulpd %ymm11,%ymm1,%ymm11
vmulpd %ymm14,%ymm0,%ymm14
vmulpd %ymm15,%ymm0,%ymm15
vfmaddsub231pd  %ymm4, %ymm0, %ymm10     # ymm10 = (ymm0 * ymm4) +/- ymm10
vfmaddsub231pd  %ymm5, %ymm0, %ymm11
vfmsubadd231pd  %ymm6, %ymm1, %ymm14
vfmsubadd231pd  %ymm7, %ymm1, %ymm15

.first_pass:
vmovupd 112(%rsi),%xmm0               /* omri   */
vinsertf128 $1, %xmm0, %ymm0, %ymm0   /* omriri */
vshufpd $15, %ymm0, %ymm0, %ymm1      /* ymm1: omiiii */
vshufpd $0,  %ymm0, %ymm0, %ymm0      /* ymm0: omrrrr */
vsubpd %ymm12,%ymm8,%ymm4
vsubpd %ymm13,%ymm9,%ymm5
vsubpd %ymm14,%ymm10,%ymm6
vsubpd %ymm15,%ymm11,%ymm7
vaddpd %ymm12,%ymm8,%ymm8
vaddpd %ymm13,%ymm9,%ymm9
vaddpd %ymm14,%ymm10,%ymm10
vaddpd %ymm15,%ymm11,%ymm11
vshufpd $5, %ymm4, %ymm4, %ymm12
vshufpd $5, %ymm5, %ymm5, %ymm13
vshufpd $5, %ymm6, %ymm6, %ymm14
vshufpd $5, %ymm7, %ymm7, %ymm15
vmulpd %ymm12,%ymm1,%ymm12
vmulpd %ymm13,%ymm1,%ymm13
vmulpd %ymm14,%ymm1,%ymm14
vmulpd %ymm15,%ymm1,%ymm15
vfmaddsub231pd  %ymm4, %ymm0, %ymm12     # ymm12 = (ymm0 * ymm4) +/- ymm12
vfmaddsub231pd  %ymm5, %ymm0, %ymm13
vfmaddsub231pd  %ymm6, %ymm0, %ymm14
vfmaddsub231pd  %ymm7, %ymm0, %ymm15

.save_and_return:
vmovupd %ymm8,(%rdi)
vmovupd %ymm9,0x20(%rdi)
vmovupd %ymm10,0x40(%rdi)
vmovupd %ymm11,0x60(%rdi)
vmovupd %ymm12,0x80(%rdi)
vmovupd %ymm13,0xa0(%rdi)
vmovupd %ymm14,0xc0(%rdi)
vmovupd %ymm15,0xe0(%rdi)
ret
