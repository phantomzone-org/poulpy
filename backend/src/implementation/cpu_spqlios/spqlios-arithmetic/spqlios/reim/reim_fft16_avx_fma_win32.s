        .text
        .p2align 4
        .globl  reim_fft16_avx_fma
        .def    reim_fft16_avx_fma;    .scl    2;      .type   32;     .endef
reim_fft16_avx_fma:

  pushq %rdi
  pushq %rsi
  movq %rcx,%rdi
  movq %rdx,%rsi
  movq %r8,%rdx
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
  callq reim_fft16_avx_fma_amd64
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

#rdi datare ptr
#rsi dataim ptr
#rdx om ptr
#.globl reim_fft16_avx_fma_amd64
reim_fft16_avx_fma_amd64:
vmovupd     (%rdi),%ymm0       # ra0
vmovupd     0x20(%rdi),%ymm1   # ra4
vmovupd     0x40(%rdi),%ymm2   # ra8
vmovupd     0x60(%rdi),%ymm3   # ra12
vmovupd     (%rsi),%ymm4       # ia0
vmovupd     0x20(%rsi),%ymm5   # ia4
vmovupd     0x40(%rsi),%ymm6   # ia8
vmovupd     0x60(%rsi),%ymm7   # ia12

vmovupd     (%rdx),%xmm12
vinsertf128 $1, %xmm12, %ymm12, %ymm12   # omriri
vshufpd     $15, %ymm12, %ymm12, %ymm13  # ymm13: omai
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # ymm12: omar
vmulpd      %ymm6,%ymm13,%ymm8 # ia0.omai
vmulpd      %ymm7,%ymm13,%ymm9 # ia4.omai
vmulpd      %ymm2,%ymm13,%ymm10 # ra0.omai
vmulpd      %ymm3,%ymm13,%ymm11 # ra4.omai
vfmsub231pd %ymm2,%ymm12,%ymm8 # rprod0
vfmsub231pd %ymm3,%ymm12,%ymm9 # rprod4
vfmadd231pd %ymm6,%ymm12,%ymm10 # iprod0
vfmadd231pd %ymm7,%ymm12,%ymm11 # iprod4
vsubpd      %ymm8,%ymm0,%ymm2
vsubpd      %ymm9,%ymm1,%ymm3
vsubpd      %ymm10,%ymm4,%ymm6
vsubpd      %ymm11,%ymm5,%ymm7
vaddpd      %ymm8,%ymm0,%ymm0
vaddpd      %ymm9,%ymm1,%ymm1
vaddpd      %ymm10,%ymm4,%ymm4
vaddpd      %ymm11,%ymm5,%ymm5

1:
vmovupd     16(%rdx),%xmm12
vinsertf128 $1, %xmm12, %ymm12, %ymm12   # omriri
vshufpd     $15, %ymm12, %ymm12, %ymm13  # ymm13: omai
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # ymm12: omar
vmulpd      %ymm5,%ymm13,%ymm8 # ia0.omai (tw)
vmulpd      %ymm7,%ymm12,%ymm9 # ia4.omar (itw)
vmulpd      %ymm1,%ymm13,%ymm10 # ra0.omai (tw)
vmulpd      %ymm3,%ymm12,%ymm11 # ra4.omar (itw)
vfmsub231pd %ymm1,%ymm12,%ymm8 # rprod0 (tw)
vfmadd231pd %ymm3,%ymm13,%ymm9 # rprod4 (itw)
vfmadd231pd %ymm5,%ymm12,%ymm10 # iprod0 (tw)
vfmsub231pd %ymm7,%ymm13,%ymm11 # iprod4 (itw)
vsubpd      %ymm8,%ymm0,%ymm1
vaddpd      %ymm9,%ymm2,%ymm3
vsubpd      %ymm10,%ymm4,%ymm5
vaddpd      %ymm11,%ymm6,%ymm7
vaddpd      %ymm8,%ymm0,%ymm0
vsubpd      %ymm9,%ymm2,%ymm2
vaddpd      %ymm10,%ymm4,%ymm4
vsubpd      %ymm11,%ymm6,%ymm6

2:
vmovupd     0x20(%rdx),%ymm12
vshufpd     $15, %ymm12, %ymm12, %ymm13  # ymm13: omaiii'i'
vshufpd     $0,  %ymm12, %ymm12, %ymm12  # ymm12: omarrr'r'

vperm2f128 $0x31,%ymm2,%ymm0,%ymm8   # ymm8 contains re to mul (tw)
vperm2f128 $0x31,%ymm3,%ymm1,%ymm9   # ymm9 contains re to mul (itw)
vperm2f128 $0x31,%ymm6,%ymm4,%ymm10   # ymm10 contains im to mul (tw)
vperm2f128 $0x31,%ymm7,%ymm5,%ymm11   # ymm11 contains im to mul (itw)
vperm2f128 $0x20,%ymm2,%ymm0,%ymm0   # ymm0 contains re to add (tw)
vperm2f128 $0x20,%ymm3,%ymm1,%ymm1   # ymm1 contains re to add (itw)
vperm2f128 $0x20,%ymm6,%ymm4,%ymm2   # ymm2 contains im to add (tw)
vperm2f128 $0x20,%ymm7,%ymm5,%ymm3   # ymm3 contains im to add (itw)

vmulpd      %ymm10,%ymm13,%ymm4 # ia0.omai (tw)
vmulpd      %ymm11,%ymm12,%ymm5 # ia4.omar (itw)
vmulpd      %ymm8,%ymm13,%ymm6 # ra0.omai (tw)
vmulpd      %ymm9,%ymm12,%ymm7 # ra4.omar (itw)
vfmsub231pd %ymm8,%ymm12,%ymm4 # rprod0 (tw)
vfmadd231pd %ymm9,%ymm13,%ymm5 # rprod4 (itw)
vfmadd231pd %ymm10,%ymm12,%ymm6 # iprod0 (tw)
vfmsub231pd %ymm11,%ymm13,%ymm7 # iprod4 (itw)
vsubpd      %ymm4,%ymm0,%ymm8
vaddpd      %ymm5,%ymm1,%ymm9
vsubpd      %ymm6,%ymm2,%ymm10
vaddpd      %ymm7,%ymm3,%ymm11
vaddpd      %ymm4,%ymm0,%ymm0
vsubpd      %ymm5,%ymm1,%ymm1
vaddpd      %ymm6,%ymm2,%ymm2
vsubpd      %ymm7,%ymm3,%ymm3

#vperm2f128 $0x31,%ymm10,%ymm2,%ymm6
#vperm2f128 $0x31,%ymm11,%ymm3,%ymm7
#vperm2f128 $0x20,%ymm10,%ymm2,%ymm4
#vperm2f128 $0x20,%ymm11,%ymm3,%ymm5
#vperm2f128 $0x31,%ymm8,%ymm0,%ymm2
#vperm2f128 $0x31,%ymm9,%ymm1,%ymm3
#vperm2f128 $0x20,%ymm8,%ymm0,%ymm0
#vperm2f128 $0x20,%ymm9,%ymm1,%ymm1

3:
vmovupd     0x40(%rdx),%ymm12
vmovupd     0x60(%rdx),%ymm13

#vperm2f128 $0x31,%ymm2,%ymm0,%ymm8   # ymm8 contains re to mul (tw)
#vperm2f128 $0x31,%ymm3,%ymm1,%ymm9   # ymm9 contains re to mul (itw)
#vperm2f128 $0x31,%ymm6,%ymm4,%ymm10   # ymm10 contains im to mul (tw)
#vperm2f128 $0x31,%ymm7,%ymm5,%ymm11   # ymm11 contains im to mul (itw)
#vperm2f128 $0x20,%ymm2,%ymm0,%ymm0   # ymm0 contains re to add (tw)
#vperm2f128 $0x20,%ymm3,%ymm1,%ymm1   # ymm1 contains re to add (itw)
#vperm2f128 $0x20,%ymm6,%ymm4,%ymm2   # ymm2 contains im to add (tw)
#vperm2f128 $0x20,%ymm7,%ymm5,%ymm3   # ymm3 contains im to add (itw)

vunpckhpd %ymm1,%ymm0,%ymm4   # (0,1) -> (0,4)
vunpckhpd %ymm3,%ymm2,%ymm6   # (2,3) -> (2,6)
vunpckhpd %ymm9,%ymm8,%ymm5   # (8,9) -> (1,5)
vunpckhpd %ymm11,%ymm10,%ymm7 # (10,11) -> (3,7)
vunpcklpd %ymm1,%ymm0,%ymm0
vunpcklpd %ymm3,%ymm2,%ymm2
vunpcklpd %ymm9,%ymm8,%ymm1
vunpcklpd %ymm11,%ymm10,%ymm3

vmulpd      %ymm6,%ymm13,%ymm8 # ia0.omai (tw)
vmulpd      %ymm7,%ymm12,%ymm9 # ia4.omar (itw)
vmulpd      %ymm4,%ymm13,%ymm10 # ra0.omai (tw)
vmulpd      %ymm5,%ymm12,%ymm11 # ra4.omar (itw)
vfmsub231pd %ymm4,%ymm12,%ymm8 # rprod0 (tw)
vfmadd231pd %ymm5,%ymm13,%ymm9 # rprod4 (itw)
vfmadd231pd %ymm6,%ymm12,%ymm10 # iprod0 (tw)
vfmsub231pd %ymm7,%ymm13,%ymm11 # iprod4 (itw)
vsubpd      %ymm8,%ymm0,%ymm4
vaddpd      %ymm9,%ymm1,%ymm5
vsubpd      %ymm10,%ymm2,%ymm6
vaddpd      %ymm11,%ymm3,%ymm7
vaddpd      %ymm8,%ymm0,%ymm0
vsubpd      %ymm9,%ymm1,%ymm1
vaddpd      %ymm10,%ymm2,%ymm2
vsubpd      %ymm11,%ymm3,%ymm3

vunpckhpd %ymm7,%ymm3,%ymm11   # (0,4) -> (0,1)
vunpckhpd %ymm5,%ymm1,%ymm9    # (2,6) -> (2,3)
vunpcklpd %ymm7,%ymm3,%ymm10
vunpcklpd %ymm5,%ymm1,%ymm8
vunpckhpd %ymm6,%ymm2,%ymm3    # (1,5) -> (8,9)
vunpckhpd %ymm4,%ymm0,%ymm1    # (3,7) -> (10,11)
vunpcklpd %ymm6,%ymm2,%ymm2
vunpcklpd %ymm4,%ymm0,%ymm0

vperm2f128 $0x31,%ymm10,%ymm2,%ymm6
vperm2f128 $0x31,%ymm11,%ymm3,%ymm7
vperm2f128 $0x20,%ymm10,%ymm2,%ymm4
vperm2f128 $0x20,%ymm11,%ymm3,%ymm5
vperm2f128 $0x31,%ymm8,%ymm0,%ymm2
vperm2f128 $0x31,%ymm9,%ymm1,%ymm3
vperm2f128 $0x20,%ymm8,%ymm0,%ymm0
vperm2f128 $0x20,%ymm9,%ymm1,%ymm1

4:
vmovupd     %ymm0,(%rdi)       # ra0
vmovupd     %ymm1,0x20(%rdi)   # ra4
vmovupd     %ymm2,0x40(%rdi)   # ra8
vmovupd     %ymm3,0x60(%rdi)   # ra12
vmovupd     %ymm4,(%rsi)       # ia0
vmovupd     %ymm5,0x20(%rsi)   # ia4
vmovupd     %ymm6,0x40(%rsi)   # ia8
vmovupd     %ymm7,0x60(%rsi)   # ia12
vzeroupper
ret
