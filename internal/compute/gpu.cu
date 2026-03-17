/*
 * gpu.cu – Full TRON address derivation pipeline on CUDA
 *
 * Pipeline per thread (one mnemonic):
 *   1. PBKDF2-HMAC-SHA512(password=mnemonic, salt="mnemonic", c=2048) → 64-byte seed
 *   2. BIP32 master key from seed  (HMAC-SHA512, key="Bitcoin seed")
 *   3. Hardened derivation: m/44'/195'/0'
 *   4. Non-hardened derivation:  /0/0
 *   5. EC multiply: pubkey = G * final_private_key  (uncompressed, 64 bytes)
 *   6. Keccak-256(pubkey) → 32-byte hash
 *   7. address = hash[12:32]  (last 20 bytes)
 *
 * Test vector:
 *   mnemonic = "afraid report escape reveal run sport pig blouse angry butter lock about"
 *   address  = 296e1e734897fc64d40b17dabd0ab4a812748542
 *
 * Build (produces libgpu_cuda.a):
 *   nvcc -O2 -arch=sm_75 --compiler-options -fPIC -c gpu.cu -o gpu.o
 *   ar rcs libgpu_cuda.a gpu.o
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include "gpu_bridge.h"

/* ================================================================
 * Byte-order helpers (SHA-512 and BIP32 use big-endian)
 * ================================================================ */
__device__ __forceinline__ uint64_t bswap64(uint64_t x) {
    uint32_t hi = (uint32_t)(x >> 32), lo = (uint32_t)x;
    hi = __byte_perm(hi, 0, 0x0123);
    lo = __byte_perm(lo, 0, 0x0123);
    return ((uint64_t)lo << 32) | hi;
}
__device__ __forceinline__ uint32_t bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}

__device__ __forceinline__ uint64_t load_be64(const uint8_t *p) {
    uint64_t v; memcpy(&v, p, 8); return bswap64(v);
}
__device__ __forceinline__ void store_be64(uint8_t *p, uint64_t v) {
    v = bswap64(v); memcpy(p, &v, 8);
}
__device__ __forceinline__ void store_be32(uint8_t *p, uint32_t v) {
    v = bswap32(v); memcpy(p, &v, 4);
}

/* ================================================================
 * SHA-512
 * ================================================================ */
__constant__ uint64_t SHA512_K[80] = {
    0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
    0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
    0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
    0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL,
};

#define ROTR64(x,n) (((x)>>(n))|((x)<<(64-(n))))
#define S0(x) (ROTR64(x,28)^ROTR64(x,34)^ROTR64(x,39))
#define S1(x) (ROTR64(x,14)^ROTR64(x,18)^ROTR64(x,41))
#define G0(x) (ROTR64(x, 1)^ROTR64(x, 8)^((x)>> 7))
#define G1(x) (ROTR64(x,19)^ROTR64(x,61)^((x)>> 6))
#define CH(x,y,z)  (((x)&(y))^(~(x)&(z)))
#define MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))

typedef struct { uint64_t h[8]; uint8_t buf[128]; uint64_t total; uint32_t blen; } sha512_ctx;

__device__ __noinline__ void sha512_compress(uint64_t h[8], const uint8_t blk[128]) {
    uint64_t W[80];
    for (int i=0;i<16;i++) W[i]=load_be64(blk+i*8);
    for (int i=16;i<80;i++) W[i]=G1(W[i-2])+W[i-7]+G0(W[i-15])+W[i-16];
    uint64_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for (int i=0;i<80;i++){
        uint64_t t1=hh+S1(e)+CH(e,f,g)+SHA512_K[i]+W[i];
        uint64_t t2=S0(a)+MAJ(a,b,c);
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
}

__device__ void sha512_init(sha512_ctx *c){
    c->h[0]=0x6a09e667f3bcc908ULL;c->h[1]=0xbb67ae8584caa73bULL;
    c->h[2]=0x3c6ef372fe94f82bULL;c->h[3]=0xa54ff53a5f1d36f1ULL;
    c->h[4]=0x510e527fade682d1ULL;c->h[5]=0x9b05688c2b3e6c1fULL;
    c->h[6]=0x1f83d9abfb41bd6bULL;c->h[7]=0x5be0cd19137e2179ULL;
    c->total=0; c->blen=0;
}

__device__ void sha512_update(sha512_ctx *c, const uint8_t *d, uint32_t n){
    while(n>0){
        uint32_t sp=128-c->blen, tk=(n<sp)?n:sp;
        memcpy(c->buf+c->blen,d,tk);
        c->blen+=tk; c->total+=tk; d+=tk; n-=tk;
        if(c->blen==128){ sha512_compress(c->h,c->buf); c->blen=0; }
    }
}

__device__ void sha512_final(sha512_ctx *c, uint8_t out[64]){
    uint64_t bits=c->total*8;
    c->buf[c->blen++]=0x80;
    if(c->blen>112){ memset(c->buf+c->blen,0,128-c->blen); sha512_compress(c->h,c->buf); c->blen=0; }
    memset(c->buf+c->blen,0,112-c->blen);
    store_be64(c->buf+112,0ULL); store_be64(c->buf+120,bits);
    sha512_compress(c->h,c->buf);
    for(int i=0;i<8;i++) store_be64(out+i*8,c->h[i]);
}

/* ================================================================
 * HMAC-SHA-512  (block=128 bytes)
 * ================================================================ */
__device__ __noinline__ void hmac_sha512(
        const uint8_t *key, uint32_t klen,
        const uint8_t *msg, uint32_t mlen,
        uint8_t out[64])
{
    uint8_t k[128]; memset(k,0,128);
    if(klen>128){ sha512_ctx t; sha512_init(&t); sha512_update(&t,key,klen); sha512_final(&t,k); }
    else memcpy(k,key,klen);

    uint8_t ipad[128],opad[128];
    for(int i=0;i<128;i++){ ipad[i]=k[i]^0x36; opad[i]=k[i]^0x5c; }

    sha512_ctx t;
    uint8_t inner[64];
    sha512_init(&t); sha512_update(&t,ipad,128); sha512_update(&t,msg,mlen); sha512_final(&t,inner);
    sha512_init(&t); sha512_update(&t,opad,128); sha512_update(&t,inner,64); sha512_final(&t,out);
}

/* ================================================================
 * PBKDF2-HMAC-SHA512  (dkLen=64, 1 block, 2048 iterations)
 * password = mnemonic bytes,  salt = "mnemonic"
 * ================================================================ */
__device__ __noinline__ void pbkdf2_hmac_sha512(
        const uint8_t *pw, uint32_t pwlen, uint8_t dk[64])
{
    /* salt || INT(1)  →  "mnemonic\x00\x00\x00\x01" */
    uint8_t sb[12] = {'m','n','e','m','o','n','i','c',0,0,0,1};
    uint8_t U[64], T[64];
    hmac_sha512(pw,pwlen,sb,12,U);
    memcpy(T,U,64);
    for(int i=1;i<2048;i++){
        hmac_sha512(pw,pwlen,U,64,U);
        for(int j=0;j<64;j++) T[j]^=U[j];
    }
    memcpy(dk,T,64);
}

/* ================================================================
 * 256-bit arithmetic, little-endian 8×uint32  (limb[0] = LSW)
 * ================================================================ */
typedef struct { uint32_t d[8]; } u256;

/* secp256k1 field prime  p = 2^256 - 2^32 - 977 */
__constant__ u256 FP = {{ 0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,
                           0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF }};
/* secp256k1 group order  n */
__constant__ u256 FN = {{ 0xD0364141,0xBFD25E8C,0xAF48A03B,0xBAAEDCE6,
                           0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF }};
/* Generator Gx, Gy (big-endian bytes) */
__constant__ uint8_t GX_BE[32]={
    0x79,0xBE,0x66,0x7E,0xF9,0xDC,0xBB,0xAC,0x55,0xA0,0x62,0x95,0xCE,0x87,0x0B,0x07,
    0x02,0x9B,0xFC,0xDB,0x2D,0xCE,0x28,0xD9,0x59,0xF2,0x81,0x5B,0x16,0xF8,0x17,0x98};
__constant__ uint8_t GY_BE[32]={
    0x48,0x3A,0xDA,0x77,0x26,0xA3,0xC4,0x65,0x5D,0xA4,0xFB,0xFC,0x0E,0x11,0x08,0xA8,
    0xFD,0x17,0xB4,0x48,0xA6,0x85,0x54,0x19,0x9C,0x47,0xD0,0x8F,0xFB,0x10,0xD4,0xB8};

__device__ void u256_from_be(u256 *r, const uint8_t b[32]){
    for(int i=0;i<8;i++){
        int o=28-i*4;
        r->d[i]=((uint32_t)b[o]<<24)|((uint32_t)b[o+1]<<16)|((uint32_t)b[o+2]<<8)|((uint32_t)b[o+3]);
    }
}
__device__ void u256_to_be(const u256 *a, uint8_t b[32]){
    for(int i=0;i<8;i++){
        uint32_t w=a->d[i]; int o=28-i*4;
        b[o]=(w>>24)&0xFF;b[o+1]=(w>>16)&0xFF;b[o+2]=(w>>8)&0xFF;b[o+3]=w&0xFF;
    }
}

__device__ __forceinline__ int u256_cmp(const u256 *a, const u256 *b){
    for(int i=7;i>=0;i--){
        if(a->d[i]<b->d[i]) return -1;
        if(a->d[i]>b->d[i]) return  1;
    }
    return 0;
}
__device__ __forceinline__ int u256_is_zero(const u256 *a){
    uint32_t r=0; for(int i=0;i<8;i++) r|=a->d[i]; return r==0;
}

/* r = a+b, returns carry */
__device__ __forceinline__ uint32_t u256_add(u256 *r, const u256 *a, const u256 *b){
    uint64_t c=0;
    for(int i=0;i<8;i++){ uint64_t s=(uint64_t)a->d[i]+b->d[i]+c; r->d[i]=(uint32_t)s; c=s>>32; }
    return (uint32_t)c;
}
/* r = a-b, returns borrow */
__device__ __forceinline__ uint32_t u256_sub(u256 *r, const u256 *a, const u256 *b){
    int64_t bw=0;
    for(int i=0;i<8;i++){ int64_t d=(int64_t)a->d[i]-b->d[i]+bw; r->d[i]=(uint32_t)d; bw=d>>32; }
    return (uint32_t)(-bw);
}

/* ================================================================
 * Field arithmetic mod p  (secp256k1 prime, a.k.a. Fp)
 * ================================================================ */
__device__ __forceinline__ void fp_reduce_once(u256 *r){
    if(u256_cmp(r,&FP)>=0){ u256 t; u256_sub(&t,r,&FP); *r=t; }
}

__device__ void fp_add(u256 *r, const u256 *a, const u256 *b){
    if(u256_add(r,a,b) || u256_cmp(r,&FP)>=0){ u256 t; u256_sub(&t,r,&FP); *r=t; }
}
__device__ void fp_sub(u256 *r, const u256 *a, const u256 *b){
    if(u256_sub(r,a,b)){ u256 t; u256_add(&t,r,&FP); *r=t; }
}
__device__ __forceinline__ void fp_sqr(u256 *r, const u256 *a);  /* fwd decl */

/*
 * Solinas reduction: p = 2^256 - 2^32 - 977, so 2^256 ≡ 2^32+977 (mod p).
 *
 * Step 1: schoolbook 8×8 → 512-bit product T[0..15] (each limb ≤ 2^32).
 * Step 2: fold T[8..15] back: T[k+8]*2^(256+32k) ≡ T[k+8]*(2^32+977)*2^(32k)
 * Step 3: one more reduction on s[8..9], then final carry clean-up.
 */
__device__ __noinline__ void fp_mul(u256 *r, const u256 *a, const u256 *b){
    /* Schoolbook 256×256 → 512-bit product in T[0..15].
     * Standard carry-propagating form keeps each T[k] ≤ 2^32-1;
     * T[8..15] each receive exactly one carry from outer iteration i=k-8. */
    uint64_t T[16];
    memset(T,0,sizeof(T));
    for(int i=0;i<8;i++){
        uint64_t carry=0;
        for(int j=0;j<8;j++){
            uint64_t uv=(uint64_t)a->d[i]*b->d[j]+T[i+j]+carry;
            T[i+j]=(uint32_t)uv; carry=uv>>32;
        }
        T[i+8]+=carry;
    }
    /* Solinas round 1: T[k+8]*2^(256+32k) ≡ T[k+8]*(2^32+977)*2^(32k) mod p */
    uint64_t s[10];
    memset(s,0,sizeof(s));
    for(int i=0;i<8;i++){
        s[i]   += T[i];
        s[i]   += T[i+8]*977;
        s[i+1] += T[i+8];
    }
    for(int i=0;i<9;i++){ s[i+1]+=s[i]>>32; s[i]&=0xFFFFFFFF; }
    /* Solinas round 2: s[8..9] are at most ~2^33 */
    uint64_t hi=s[8]+(s[9]<<32); s[8]=0; s[9]=0;
    s[0]+=hi*977; s[1]+=hi;
    for(int i=0;i<8;i++){ s[i+1]+=s[i]>>32; s[i]&=0xFFFFFFFF; }
    /* Residual carry in s[8] (≤1): 2^256 ≡ 2^32+977 mod p */
    if(s[8]){ s[0]+=977; s[1]+=1; for(int i=0;i<8;i++){ s[i+1]+=s[i]>>32; s[i]&=0xFFFFFFFF; } }
    for(int i=0;i<8;i++) r->d[i]=(uint32_t)s[i];
    fp_reduce_once(r);
    fp_reduce_once(r);
}

__device__ __forceinline__ void fp_sqr(u256 *r, const u256 *a){ fp_mul(r,a,a); }

/* Modular inverse: a^(p-2) mod p via square-and-multiply */
__device__ __noinline__ void fp_inv(u256 *r, const u256 *a){
    /* p-2 has the same bit pattern as p except last 2 bits: ...FC2D */
    /* bit 0 = 1, bit 1 = 0, bits 2..31 = same as p-0 = ...FC2F minus 2 */
    /* Just use p with exp = p-2 computed at runtime */
    u256 exp = FP;          /* copy of p */
    exp.d[0] -= 2;          /* exp = p - 2 */

    u256 base = *a;
    memset(r,0,sizeof(*r)); r->d[0]=1;  /* r = 1 */

    for(int bit=255;bit>=0;bit--){
        fp_sqr(r,r);
        if((exp.d[bit>>5]>>(bit&31))&1) fp_mul(r,r,&base);
    }
}

/* ================================================================
 * Scalar field arithmetic mod n
 * ================================================================ */
__device__ void fn_add(u256 *r, const u256 *a, const u256 *b){
    if(u256_add(r,a,b) || u256_cmp(r,&FN)>=0){ u256 t; u256_sub(&t,r,&FN); *r=t; }
}

/* ================================================================
 * secp256k1 EC points (Jacobian X:Y:Z, affine = X/Z², Y/Z³)
 * ================================================================ */
typedef struct { u256 X,Y,Z; } jpoint;
typedef struct { u256 x,y;   } apoint;

__device__ __forceinline__ int jp_is_inf(const jpoint *p){ return u256_is_zero(&p->Z); }

__device__ __noinline__ void jp_to_affine(apoint *r, const jpoint *p){
    u256 zi,zi2,zi3;
    fp_inv(&zi,&p->Z);
    fp_sqr(&zi2,&zi);
    fp_mul(&zi3,&zi2,&zi);
    fp_mul(&r->x,&p->X,&zi2);
    fp_mul(&r->y,&p->Y,&zi3);
}

/*
 * Point doubling (secp256k1, a=0) – formula dbl-2009-l:
 *  A=X1²  B=Y1²  C=B²  D=2((X1+B)²-A-C)  E=3A  F=E²
 *  X3=F-2D  Y3=E(D-X3)-8C  Z3=2Y1Z1
 */
__device__ __noinline__ void jp_double(jpoint *r, const jpoint *p){
    if(jp_is_inf(p)){*r=*p;return;}
    u256 A,B,C,D,E,F,t;
    fp_sqr(&A,&p->X);
    fp_sqr(&B,&p->Y);
    fp_sqr(&C,&B);
    fp_add(&t,&p->X,&B); fp_sqr(&D,&t); fp_sub(&D,&D,&A); fp_sub(&D,&D,&C); fp_add(&D,&D,&D);
    fp_add(&E,&A,&A); fp_add(&E,&E,&A);
    fp_sqr(&F,&E);
    /* Z3 = 2*Y1*Z1 must be computed before Y1 is overwritten by Y3 (aliasing: r==p) */
    fp_mul(&r->Z,&p->Y,&p->Z); fp_add(&r->Z,&r->Z,&r->Z);
    fp_add(&t,&D,&D); fp_sub(&r->X,&F,&t);
    fp_sub(&t,&D,&r->X); fp_mul(&r->Y,&E,&t);
    u256 eC; fp_add(&eC,&C,&C);fp_add(&eC,&eC,&eC);fp_add(&eC,&eC,&eC);
    fp_sub(&r->Y,&r->Y,&eC);
}

/*
 * Point addition (Jacobian + Jacobian) – formula add-2007-bl
 */
__device__ __noinline__ void jp_add(jpoint *r, const jpoint *p, const jpoint *q){
    if(jp_is_inf(p)){*r=*q;return;}
    if(jp_is_inf(q)){*r=*p;return;}
    u256 Z1Z1,Z2Z2,U1,U2,S1,S2,H,I,J,rv,V,t;
    fp_sqr(&Z1Z1,&p->Z); fp_sqr(&Z2Z2,&q->Z);
    fp_mul(&U1,&p->X,&Z2Z2); fp_mul(&U2,&q->X,&Z1Z1);
    fp_mul(&t,&p->Y,&q->Z); fp_mul(&S1,&t,&Z2Z2);
    fp_mul(&t,&q->Y,&p->Z); fp_mul(&S2,&t,&Z1Z1);
    fp_sub(&H,&U2,&U1);
    if(u256_is_zero(&H)){
        /* Degenerate: same X, check Y */
        u256 negS1; fp_sub(&negS1,&FP,&S1); fp_reduce_once(&negS1);
        if(u256_cmp(&S2,&negS1)==0){ memset(r,0,sizeof(*r)); return; } /* P == -Q → inf */
        jp_double(r,p); return;
    }
    fp_add(&I,&H,&H); fp_sqr(&I,&I);
    fp_mul(&J,&H,&I);
    fp_sub(&rv,&S2,&S1); fp_add(&rv,&rv,&rv);
    fp_mul(&V,&U1,&I);
    fp_sqr(&r->X,&rv); fp_sub(&r->X,&r->X,&J); fp_sub(&r->X,&r->X,&V); fp_sub(&r->X,&r->X,&V);
    fp_sub(&t,&V,&r->X); fp_mul(&r->Y,&rv,&t);
    fp_mul(&t,&S1,&J); fp_add(&t,&t,&t); fp_sub(&r->Y,&r->Y,&t);
    fp_add(&t,&p->Z,&q->Z); fp_sqr(&t,&t);
    fp_sub(&t,&t,&Z1Z1); fp_sub(&t,&t,&Z2Z2); fp_mul(&r->Z,&t,&H);
}

/* ================================================================
 * Precomputed table: GT[i] = (i+1)*G, i=0..14, affine big-endian bytes
 * ================================================================ */
typedef struct { uint8_t x[32]; uint8_t y[32]; } apoint_bytes;
__constant__ apoint_bytes GT[15] = {
    /* 1G */  {{0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,0x07,0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98},{0x48,0x3a,0xda,0x77,0x26,0xa3,0xc4,0x65,0x5d,0xa4,0xfb,0xfc,0x0e,0x11,0x08,0xa8,0xfd,0x17,0xb4,0x48,0xa6,0x85,0x54,0x19,0x9c,0x47,0xd0,0x8f,0xfb,0x10,0xd4,0xb8}},
    /* 2G */  {{0xc6,0x04,0x7f,0x94,0x41,0xed,0x7d,0x6d,0x30,0x45,0x40,0x6e,0x95,0xc0,0x7c,0xd8,0x5c,0x77,0x8e,0x4b,0x8c,0xef,0x3c,0xa7,0xab,0xac,0x09,0xb9,0x5c,0x70,0x9e,0xe5},{0x1a,0xe1,0x68,0xfe,0xa6,0x3d,0xc3,0x39,0xa3,0xc5,0x84,0x19,0x46,0x6c,0xea,0xee,0xf7,0xf6,0x32,0x65,0x32,0x66,0xd0,0xe1,0x23,0x64,0x31,0xa9,0x50,0xcf,0xe5,0x2a}},
    /* 3G */  {{0xf9,0x30,0x8a,0x01,0x92,0x58,0xc3,0x10,0x49,0x34,0x4f,0x85,0xf8,0x9d,0x52,0x29,0xb5,0x31,0xc8,0x45,0x83,0x6f,0x99,0xb0,0x86,0x01,0xf1,0x13,0xbc,0xe0,0x36,0xf9},{0x38,0x8f,0x7b,0x0f,0x63,0x2d,0xe8,0x14,0x0f,0xe3,0x37,0xe6,0x2a,0x37,0xf3,0x56,0x65,0x00,0xa9,0x99,0x34,0xc2,0x23,0x1b,0x6c,0xb9,0xfd,0x75,0x84,0xb8,0xe6,0x72}},
    /* 4G */  {{0xe4,0x93,0xdb,0xf1,0xc1,0x0d,0x80,0xf3,0x58,0x1e,0x49,0x04,0x93,0x0b,0x14,0x04,0xcc,0x6c,0x13,0x90,0x0e,0xe0,0x75,0x84,0x74,0xfa,0x94,0xab,0xe8,0xc4,0xcd,0x13},{0x51,0xed,0x99,0x3e,0xa0,0xd4,0x55,0xb7,0x56,0x42,0xe2,0x09,0x8e,0xa5,0x14,0x48,0xd9,0x67,0xae,0x33,0xbf,0xbd,0xfe,0x40,0xcf,0xe9,0x7b,0xdc,0x47,0x73,0x99,0x22}},
    /* 5G */  {{0x2f,0x8b,0xde,0x4d,0x1a,0x07,0x20,0x93,0x55,0xb4,0xa7,0x25,0x0a,0x5c,0x51,0x28,0xe8,0x8b,0x84,0xbd,0xdc,0x61,0x9a,0xb7,0xcb,0xa8,0xd5,0x69,0xb2,0x40,0xef,0xe4},{0xd8,0xac,0x22,0x26,0x36,0xe5,0xe3,0xd6,0xd4,0xdb,0xa9,0xdd,0xa6,0xc9,0xc4,0x26,0xf7,0x88,0x27,0x1b,0xab,0x0d,0x68,0x40,0xdc,0xa8,0x7d,0x3a,0xa6,0xac,0x62,0xd6}},
    /* 6G */  {{0xff,0xf9,0x7b,0xd5,0x75,0x5e,0xee,0xa4,0x20,0x45,0x3a,0x14,0x35,0x52,0x35,0xd3,0x82,0xf6,0x47,0x2f,0x85,0x68,0xa1,0x8b,0x2f,0x05,0x7a,0x14,0x60,0x29,0x75,0x56},{0xae,0x12,0x77,0x7a,0xac,0xfb,0xb6,0x20,0xf3,0xbe,0x96,0x01,0x7f,0x45,0xc5,0x60,0xde,0x80,0xf0,0xf6,0x51,0x8f,0xe4,0xa0,0x3c,0x87,0x0c,0x36,0xb0,0x75,0xf2,0x97}},
    /* 7G */  {{0x5c,0xbd,0xf0,0x64,0x6e,0x5d,0xb4,0xea,0xa3,0x98,0xf3,0x65,0xf2,0xea,0x7a,0x0e,0x3d,0x41,0x9b,0x7e,0x03,0x30,0xe3,0x9c,0xe9,0x2b,0xdd,0xed,0xca,0xc4,0xf9,0xbc},{0x6a,0xeb,0xca,0x40,0xba,0x25,0x59,0x60,0xa3,0x17,0x8d,0x6d,0x86,0x1a,0x54,0xdb,0xa8,0x13,0xd0,0xb8,0x13,0xfd,0xe7,0xb5,0xa5,0x08,0x26,0x28,0x08,0x72,0x64,0xda}},
    /* 8G */  {{0x2f,0x01,0xe5,0xe1,0x5c,0xca,0x35,0x1d,0xaf,0xf3,0x84,0x3f,0xb7,0x0f,0x3c,0x2f,0x0a,0x1b,0xdd,0x05,0xe5,0xaf,0x88,0x8a,0x67,0x78,0x4e,0xf3,0xe1,0x0a,0x2a,0x01},{0x5c,0x4d,0xa8,0xa7,0x41,0x53,0x99,0x49,0x29,0x3d,0x08,0x2a,0x13,0x2d,0x13,0xb4,0xc2,0xe2,0x13,0xd6,0xba,0x5b,0x76,0x17,0xb5,0xda,0x2c,0xb7,0x6c,0xbd,0xe9,0x04}},
    /* 9G */  {{0xac,0xd4,0x84,0xe2,0xf0,0xc7,0xf6,0x53,0x09,0xad,0x17,0x8a,0x9f,0x55,0x9a,0xbd,0xe0,0x97,0x96,0x97,0x4c,0x57,0xe7,0x14,0xc3,0x5f,0x11,0x0d,0xfc,0x27,0xcc,0xbe},{0xcc,0x33,0x89,0x21,0xb0,0xa7,0xd9,0xfd,0x64,0x38,0x09,0x71,0x76,0x3b,0x61,0xe9,0xad,0xd8,0x88,0xa4,0x37,0x5f,0x8e,0x0f,0x05,0xcc,0x26,0x2a,0xc6,0x4f,0x9c,0x37}},
    /* 10G */ {{0xa0,0x43,0x4d,0x9e,0x47,0xf3,0xc8,0x62,0x35,0x47,0x7c,0x7b,0x1a,0xe6,0xae,0x5d,0x34,0x42,0xd4,0x9b,0x19,0x43,0xc2,0xb7,0x52,0xa6,0x8e,0x2a,0x47,0xe2,0x47,0xc7},{0x89,0x3a,0xba,0x42,0x54,0x19,0xbc,0x27,0xa3,0xb6,0xc7,0xe6,0x93,0xa2,0x4c,0x69,0x6f,0x79,0x4c,0x2e,0xd8,0x77,0xa1,0x59,0x3c,0xbe,0xe5,0x3b,0x03,0x73,0x68,0xd7}},
    /* 11G */ {{0x77,0x4a,0xe7,0xf8,0x58,0xa9,0x41,0x1e,0x5e,0xf4,0x24,0x6b,0x70,0xc6,0x5a,0xac,0x56,0x49,0x98,0x0b,0xe5,0xc1,0x78,0x91,0xbb,0xec,0x17,0x89,0x5d,0xa0,0x08,0xcb},{0xd9,0x84,0xa0,0x32,0xeb,0x6b,0x5e,0x19,0x02,0x43,0xdd,0x56,0xd7,0xb7,0xb3,0x65,0x37,0x2d,0xb1,0xe2,0xdf,0xf9,0xd6,0xa8,0x30,0x1d,0x74,0xc9,0xc9,0x53,0xc6,0x1b}},
    /* 12G */ {{0xd0,0x11,0x15,0xd5,0x48,0xe7,0x56,0x1b,0x15,0xc3,0x8f,0x00,0x4d,0x73,0x46,0x33,0x68,0x7c,0xf4,0x41,0x96,0x20,0x09,0x5b,0xc5,0xb0,0xf4,0x70,0x70,0xaf,0xe8,0x5a},{0xa9,0xf3,0x4f,0xfd,0xc8,0x15,0xe0,0xd7,0xa8,0xb6,0x45,0x37,0xe1,0x7b,0xd8,0x15,0x79,0x23,0x8c,0x5d,0xd9,0xa8,0x6d,0x52,0x6b,0x05,0x1b,0x13,0xf4,0x06,0x23,0x27}},
    /* 13G */ {{0xf2,0x87,0x73,0xc2,0xd9,0x75,0x28,0x8b,0xc7,0xd1,0xd2,0x05,0xc3,0x74,0x86,0x51,0xb0,0x75,0xfb,0xc6,0x61,0x0e,0x58,0xcd,0xde,0xed,0xdf,0x8f,0x19,0x40,0x5a,0xa8},{0x0a,0xb0,0x90,0x2e,0x8d,0x88,0x0a,0x89,0x75,0x82,0x12,0xeb,0x65,0xcd,0xaf,0x47,0x3a,0x1a,0x06,0xda,0x52,0x1f,0xa9,0x1f,0x29,0xb5,0xcb,0x52,0xdb,0x03,0xed,0x81}},
    /* 14G */ {{0x49,0x9f,0xdf,0x9e,0x89,0x5e,0x71,0x9c,0xfd,0x64,0xe6,0x7f,0x07,0xd3,0x8e,0x32,0x26,0xaa,0x7b,0x63,0x67,0x89,0x49,0xe6,0xe4,0x9b,0x24,0x1a,0x60,0xe8,0x23,0xe4},{0xca,0xc2,0xf6,0xc4,0xb5,0x4e,0x85,0x51,0x90,0xf0,0x44,0xe4,0xa7,0xb3,0xd4,0x64,0x46,0x42,0x79,0xc2,0x7a,0x3f,0x95,0xbc,0xc6,0x5f,0x40,0xd4,0x03,0xa1,0x3f,0x5b}},
    /* 15G */ {{0xd7,0x92,0x4d,0x4f,0x7d,0x43,0xea,0x96,0x5a,0x46,0x5a,0xe3,0x09,0x5f,0xf4,0x11,0x31,0xe5,0x94,0x6f,0x3c,0x85,0xf7,0x9e,0x44,0xad,0xbc,0xf8,0xe2,0x7e,0x08,0x0e},{0x58,0x1e,0x28,0x72,0xa8,0x6c,0x72,0xa6,0x83,0x84,0x2e,0xc2,0x28,0xcc,0x6d,0xef,0xea,0x40,0xaf,0x2b,0xd8,0x96,0xd3,0xa5,0xc5,0x04,0xdc,0x9f,0xf6,0xa2,0x6b,0x58}}
};

/*
 * Mixed Jacobian+Affine point addition (Z2=1).
 * Cost: 3 sqr + 7 mul  (vs jp_add: 6 sqr + 8 mul).
 * Formula: madd-2007-bl (Z2=1 specialisation).
 */
__device__ __noinline__ void jp_madd(jpoint *r, const jpoint *p, const u256 *x2, const u256 *y2){
    if(jp_is_inf(p)){ r->X=*x2; r->Y=*y2; memset(&r->Z,0,sizeof(r->Z)); r->Z.d[0]=1; return; }
    u256 Z1Z1,U2,Z1_3,S2,H,HH,I,J,rv,V,t;
    fp_sqr(&Z1Z1,&p->Z);                         /* Z1Z1 = Z1^2        */
    fp_mul(&U2,x2,&Z1Z1);                         /* U2   = X2*Z1Z1     */
    fp_mul(&Z1_3,&p->Z,&Z1Z1);                    /* Z1^3               */
    fp_mul(&S2,y2,&Z1_3);                         /* S2   = Y2*Z1^3     */
    fp_sub(&H,&U2,&p->X);                         /* H    = U2 - X1     */
    if(u256_is_zero(&H)){                          /* degenerate: X equal */
        u256 negY1; fp_sub(&negY1,&FP,&p->Y); fp_reduce_once(&negY1);
        if(u256_cmp(&S2,&negY1)==0){ memset(r,0,sizeof(*r)); return; }
        jp_double(r,p); return;
    }
    fp_sqr(&HH,&H);                               /* HH   = H^2         */
    fp_add(&I,&HH,&HH); fp_add(&I,&I,&I);        /* I    = 4*HH        */
    fp_mul(&J,&H,&I);                             /* J    = H*I = 4H^3  */
    fp_sub(&rv,&S2,&p->Y); fp_add(&rv,&rv,&rv);  /* r    = 2*(S2-Y1)   */
    fp_mul(&V,&p->X,&I);                          /* V    = X1*I        */
    fp_sqr(&r->X,&rv);
    fp_sub(&r->X,&r->X,&J); fp_sub(&r->X,&r->X,&V); fp_sub(&r->X,&r->X,&V);
    fp_sub(&t,&V,&r->X); fp_mul(&r->Y,&rv,&t);
    fp_mul(&t,&p->Y,&J); fp_add(&t,&t,&t); fp_sub(&r->Y,&r->Y,&t);
    fp_mul(&r->Z,&p->Z,&H); fp_add(&r->Z,&r->Z,&r->Z); /* Z3 = 2*Z1*H */
}

/* Scalar × G — binary double-and-add */
__device__ __noinline__ void ec_mul_G(jpoint *r, const u256 *k){
    jpoint G;
    u256_from_be(&G.X, GX_BE);
    u256_from_be(&G.Y, GY_BE);
    memset(&G.Z, 0, sizeof(G.Z));
    G.Z.d[0] = 1;
    memset(r, 0, sizeof(*r));   /* infinity: Z=0 */
    for(int bit = 255; bit >= 0; bit--){
        jp_double(r, r);
        if((k->d[bit>>5] >> (bit&31)) & 1)
            jp_add(r, r, &G);
    }
}

/* ================================================================
 * BIP32 key derivation
 * ================================================================ */
typedef struct { uint8_t key[32]; uint8_t cc[32]; } bip32key;

__device__ void bip32_master(bip32key *out, const uint8_t seed[64]){
    const uint8_t bseed[12]={'B','i','t','c','o','i','n',' ','s','e','e','d'};
    uint8_t I[64];
    hmac_sha512(bseed,12,seed,64,I);
    memcpy(out->key,I,32); memcpy(out->cc,I+32,32);
}

/* compressed pubkey of priv (33 bytes: 02/03 || X) */
__device__ __noinline__ void priv_to_cpub(const uint8_t priv[32], uint8_t pub[33]){
    u256 k; u256_from_be(&k,priv);
    jpoint P; ec_mul_G(&P,&k);
    apoint A; jp_to_affine(&A,&P);
    pub[0]=(A.y.d[0]&1)?0x03:0x02;
    u256_to_be(&A.x,pub+1);
}

/* uncompressed pubkey (64 bytes: X||Y, no 0x04 prefix) */
__device__ __noinline__ void priv_to_upub64(const uint8_t priv[32], uint8_t pub[64]){
    u256 k; u256_from_be(&k,priv);
    jpoint P; ec_mul_G(&P,&k);
    apoint A; jp_to_affine(&A,&P);
    u256_to_be(&A.x,pub);
    u256_to_be(&A.y,pub+32);
}

__device__ __noinline__ void bip32_child_hard(bip32key *out, const bip32key *par, uint32_t idx){
    uint8_t data[37]; data[0]=0x00;
    memcpy(data+1,par->key,32);
    data[33]=(idx>>24)&0xFF; data[34]=(idx>>16)&0xFF;
    data[35]=(idx>> 8)&0xFF; data[36]=(idx    )&0xFF;
    uint8_t I[64];
    hmac_sha512(par->cc,32,data,37,I);
    u256 IL,pk,child;
    u256_from_be(&IL,I); u256_from_be(&pk,par->key);
    fn_add(&child,&IL,&pk);
    u256_to_be(&child,out->key); memcpy(out->cc,I+32,32);
}

__device__ __noinline__ void bip32_child_norm(bip32key *out, const bip32key *par, uint32_t idx){
    uint8_t data[37];
    priv_to_cpub(par->key,data);   /* 33 bytes compressed pubkey */
    data[33]=(idx>>24)&0xFF; data[34]=(idx>>16)&0xFF;
    data[35]=(idx>> 8)&0xFF; data[36]=(idx    )&0xFF;
    uint8_t I[64];
    hmac_sha512(par->cc,32,data,37,I);
    u256 IL,pk,child;
    u256_from_be(&IL,I); u256_from_be(&pk,par->key);
    fn_add(&child,&IL,&pk);
    u256_to_be(&child,out->key); memcpy(out->cc,I+32,32);
}

/* m/44'/195'/0'/0/0 */
__device__ __noinline__ void bip44_tron(const uint8_t seed[64], uint8_t privkey[32]){
    bip32key m,k1,k2,k3,k4,k5;
    bip32_master(&m,seed);
    bip32_child_hard(&k1,&m, 0x80000000u+44);
    bip32_child_hard(&k2,&k1,0x80000000u+195);
    bip32_child_hard(&k3,&k2,0x80000000u+0);
    bip32_child_norm (&k4,&k3,0);
    bip32_child_norm (&k5,&k4,0);
    memcpy(privkey,k5.key,32);
}

/* ================================================================
 * Keccak-256 (Ethereum variant: padding byte 0x01, rate=136 bytes)
 * ================================================================ */
#define KRATE 136
#define NROUNDS 24

__constant__ uint64_t KRC[24]={
    0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
    0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
    0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
    0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
    0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
    0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL,
};
__constant__ int KRHO[25]={ 0, 1,62,28,27,36,44, 6,55,20, 3,10,43,25,39,41,45,15,21, 8,18, 2,61,56,14 };
__constant__ int KPI[25] ={ 0,10,20, 5,15,16, 1,11,21, 6, 7,17, 2,12,22,23, 8,18, 3,13,14,24, 9,19, 4 };

__device__ __noinline__ void keccak_f1600(uint64_t A[25]){
    for(int r=0;r<NROUNDS;r++){
        uint64_t C[5],D[5];
        for(int x=0;x<5;x++) C[x]=A[x]^A[x+5]^A[x+10]^A[x+15]^A[x+20];
        for(int x=0;x<5;x++) D[x]=C[(x+4)%5]^ROTR64(C[(x+1)%5],63); /* ROL 1 */
        for(int i=0;i<25;i++) A[i]^=D[i%5];
        uint64_t B[25];
        for(int i=0;i<25;i++){
            int rho=KRHO[i];
            B[KPI[i]]=rho? ROTR64(A[i],64-rho) : A[i];
        }
        for(int i=0;i<25;i++) A[i]=B[i]^(~B[(i/5)*5+(i%5+1)%5]&B[(i/5)*5+(i%5+2)%5]);
        A[0]^=KRC[r];
    }
}

__device__ __noinline__ void keccak256(const uint8_t *data, uint32_t len, uint8_t out[32]){
    uint64_t st[25]; memset(st,0,200);
    uint32_t off=0;
    while(len-off>=KRATE){
        for(int i=0;i<KRATE/8;i++){ uint64_t w; memcpy(&w,data+off+i*8,8); st[i]^=w; }
        keccak_f1600(st); off+=KRATE;
    }
    uint8_t last[KRATE]; memset(last,0,KRATE);
    uint32_t rem=len-off;
    memcpy(last,data+off,rem);
    last[rem]=0x01;
    last[KRATE-1]^=0x80;
    for(int i=0;i<KRATE/8;i++){ uint64_t w; memcpy(&w,last+i*8,8); st[i]^=w; }
    keccak_f1600(st);
    memcpy(out,st,32);
}

/* ================================================================
 * Kernel: one thread per mnemonic
 * ================================================================ */
__global__ void tron_kernel(
        const uint8_t *mdata, const int *moff, const int *mlen,
        int count, uint8_t *out)
{
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx>=count) return;

    const uint8_t *mn=mdata+moff[idx];
    uint32_t       ml=(uint32_t)mlen[idx];

    uint8_t seed[64];
    pbkdf2_hmac_sha512(mn,ml,seed);

    uint8_t priv[32];
    bip44_tron(seed,priv);

    uint8_t pub[64];
    priv_to_upub64(priv,pub);

    uint8_t khash[32];
    keccak256(pub,64,khash);

    memcpy(out+idx*20, khash+12, 20);
}

/* ================================================================
 * Host functions
 * ================================================================ */
extern "C" {

int gpu_device_count(void){
    int n=0;
    cudaDeviceSetLimit(cudaLimitStackSize, 65536);
    cudaGetDeviceCount(&n);
    return n;
}

int gpu_compute_addresses(
        int            device_id,
        const uint8_t *mnemonic_data,
        const int     *mnemonic_offsets,
        const int     *mnemonic_lengths,
        int            count,
        uint8_t       *addresses_out)
{
    if(cudaSetDevice(device_id)!=cudaSuccess) return -1;
    if(count<=0) return 0;

    int max_data=0;
    for(int i=0;i<count;i++){
        int e=mnemonic_offsets[i]+mnemonic_lengths[i];
        if(e>max_data) max_data=e;
    }
    if(max_data<=0) max_data=1;

    uint8_t *dd=NULL; int *doff=NULL,*dlen=NULL; uint8_t *dout=NULL;
    if(cudaMalloc(&dd,  max_data          )!=cudaSuccess) goto err;
    if(cudaMalloc(&doff,count*sizeof(int) )!=cudaSuccess) goto err;
    if(cudaMalloc(&dlen,count*sizeof(int) )!=cudaSuccess) goto err;
    if(cudaMalloc(&dout,count*20          )!=cudaSuccess) goto err;

    cudaMemcpy(dd,  mnemonic_data,    max_data,           cudaMemcpyHostToDevice);
    cudaMemcpy(doff,mnemonic_offsets, count*sizeof(int),  cudaMemcpyHostToDevice);
    cudaMemcpy(dlen,mnemonic_lengths, count*sizeof(int),  cudaMemcpyHostToDevice);

    {
        /* Default CUDA thread stack is 1024 bytes; our kernel nests deeply:
         * pbkdf2→hmac→sha512_compress(W[80]=640B) + bip44→ec_mul_G→jp_add(12×u256)+fp_mul(T[16]+s[10])
         * Estimated peak stack ~2500B per thread – must increase before launch. */
        cudaDeviceSetLimit(cudaLimitStackSize, 65536);
        int bs=32, nb=(count+bs-1)/bs;
        tron_kernel<<<nb,bs>>>(dd,doff,dlen,count,dout);
        if(cudaDeviceSynchronize()!=cudaSuccess) goto err;
    }

    cudaMemcpy(addresses_out,dout,count*20,cudaMemcpyDeviceToHost);
    cudaFree(dd);cudaFree(doff);cudaFree(dlen);cudaFree(dout);
    return 0;
err:
    if(dd)   cudaFree(dd);
    if(doff) cudaFree(doff);
    if(dlen) cudaFree(dlen);
    if(dout) cudaFree(dout);
    return -1;
}

} /* extern "C" */
