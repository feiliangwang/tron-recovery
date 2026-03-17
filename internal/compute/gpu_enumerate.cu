/*
 * gpu_enumerate.cu – Self-contained BIP39 GPU enumeration kernel
 *
 * Compiled as an INDEPENDENT translation unit (plain -c, NOT --device-c).
 * All device functions are duplicated here to avoid CUDA RDC fat-binary
 * registration issues when linking via CGO (gcc-based linker).
 *
 * Without RDC, each .cu file gets its own complete device code image.
 * ptxas sees only the kernels in THIS file, so tron_enumerate_kernel's
 * heavy register usage cannot interfere with tron_kernel in gpu.cu.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>
#include "gpu_bridge.h"
#include "bip39_wordlist.h"

/* ================================================================
 * Byte-order helpers
 * ================================================================ */
__device__ __forceinline__ uint64_t ec_bswap64(uint64_t x) {
    uint32_t hi = (uint32_t)(x >> 32), lo = (uint32_t)x;
    hi = __byte_perm(hi, 0, 0x0123);
    lo = __byte_perm(lo, 0, 0x0123);
    return ((uint64_t)lo << 32) | hi;
}
__device__ __forceinline__ uint32_t ec_bswap32(uint32_t x) {
    return __byte_perm(x, 0, 0x0123);
}
__device__ __forceinline__ uint64_t ec_load_be64(const uint8_t *p) {
    uint64_t v; memcpy(&v, p, 8); return ec_bswap64(v);
}
__device__ __forceinline__ void ec_store_be64(uint8_t *p, uint64_t v) {
    v = ec_bswap64(v); memcpy(p, &v, 8);
}
__device__ __forceinline__ void ec_store_be32(uint8_t *p, uint32_t v) {
    v = ec_bswap32(v); memcpy(p, &v, 4);
}

/* ================================================================
 * SHA-512
 * ================================================================ */
__constant__ uint64_t EN_SHA512_K[80] = {
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

#define EN_ROTR64(x,n) (((x)>>(n))|((x)<<(64-(n))))
#define EN_S0(x) (EN_ROTR64(x,28)^EN_ROTR64(x,34)^EN_ROTR64(x,39))
#define EN_S1(x) (EN_ROTR64(x,14)^EN_ROTR64(x,18)^EN_ROTR64(x,41))
#define EN_G0(x) (EN_ROTR64(x, 1)^EN_ROTR64(x, 8)^((x)>> 7))
#define EN_G1(x) (EN_ROTR64(x,19)^EN_ROTR64(x,61)^((x)>> 6))
#define EN_CH(x,y,z)  (((x)&(y))^(~(x)&(z)))
#define EN_MAJ(x,y,z) (((x)&(y))^((x)&(z))^((y)&(z)))

typedef struct { uint64_t h[8]; uint8_t buf[128]; uint64_t total; uint32_t blen; } en_sha512_ctx;

__device__ __noinline__ void en_sha512_compress(uint64_t h[8], const uint8_t blk[128]) {
    /* W[16] circular buffer: 128 B stack vs 640 B with W[80].
     * i>=16: W[i&15] is overwritten in-place (was W[i-16]), result is W[i]. */
    uint64_t W[16];
    for (int i = 0; i < 16; i++) W[i] = ec_load_be64(blk + i * 8);
    uint64_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll 8
    for (int i = 0; i < 80; i++) {
        if (i >= 16)
            W[i&15] = EN_G1(W[(i-2)&15]) + W[(i-7)&15] + EN_G0(W[(i-15)&15]) + W[i&15];
        uint64_t t1 = hh + EN_S1(e) + EN_CH(e,f,g) + EN_SHA512_K[i] + W[i&15];
        uint64_t t2 = EN_S0(a) + EN_MAJ(a,b,c);
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
}

__device__ void en_sha512_init(en_sha512_ctx *c){
    c->h[0]=0x6a09e667f3bcc908ULL;c->h[1]=0xbb67ae8584caa73bULL;
    c->h[2]=0x3c6ef372fe94f82bULL;c->h[3]=0xa54ff53a5f1d36f1ULL;
    c->h[4]=0x510e527fade682d1ULL;c->h[5]=0x9b05688c2b3e6c1fULL;
    c->h[6]=0x1f83d9abfb41bd6bULL;c->h[7]=0x5be0cd19137e2179ULL;
    c->total=0; c->blen=0;
}

__device__ void en_sha512_update(en_sha512_ctx *c, const uint8_t *d, uint32_t n){
    while(n>0){
        uint32_t sp=128-c->blen, tk=(n<sp)?n:sp;
        memcpy(c->buf+c->blen,d,tk);
        c->blen+=tk; c->total+=tk; d+=tk; n-=tk;
        if(c->blen==128){ en_sha512_compress(c->h,c->buf); c->blen=0; }
    }
}

__device__ void en_sha512_final(en_sha512_ctx *c, uint8_t out[64]){
    uint64_t bits=c->total*8;
    c->buf[c->blen++]=0x80;
    if(c->blen>112){ memset(c->buf+c->blen,0,128-c->blen); en_sha512_compress(c->h,c->buf); c->blen=0; }
    memset(c->buf+c->blen,0,112-c->blen);
    ec_store_be64(c->buf+112,0ULL); ec_store_be64(c->buf+120,bits);
    en_sha512_compress(c->h,c->buf);
    for(int i=0;i<8;i++) ec_store_be64(out+i*8,c->h[i]);
}

/* ================================================================
 * sha512_resume_64 – zero-spill PBKDF2 inner/outer hash
 *
 * Computes SHA-512(state || data64), where `state` is the hash
 * state after processing exactly 128 bytes (one full block).
 * Total message = 128 + 64 = 192 bytes = 1536 bits.
 *
 * Replaces the expensive `en_sha512_ctx t = ctx_copy` pattern
 * that causes 664 bytes of Local Memory spills per pbkdf2 call.
 * The W schedule is built entirely in registers (no block[128]).
 * ================================================================ */
__device__ __forceinline__ void sha512_resume_64(
    const uint64_t st[8], const uint8_t d[64], uint8_t out[64])
{
    uint64_t h0=st[0],h1=st[1],h2=st[2],h3=st[3];
    uint64_t h4=st[4],h5=st[5],h6=st[6],h7=st[7];

    /* W[0..7]: 64 bytes of data (big-endian 64-bit words)          */
    /* W[8]   = 0x80 padding bit at byte offset 64                  */
    /* W[9..13] = 0                                                 */
    /* W[14]  = 0   (high 64 bits of bit-length)                    */
    /* W[15]  = 1536 (192 bytes × 8 bits)                           */
    uint64_t W[16];
    for (int i = 0; i < 8; i++) W[i] = ec_load_be64(d + i * 8);
    W[8]  = 0x8000000000000000ULL;
    W[9]  = 0; W[10] = 0; W[11] = 0; W[12] = 0; W[13] = 0;
    W[14] = 0; W[15] = 1536ULL;

    uint64_t a=h0,b=h1,c=h2,d_=h3,e=h4,f=h5,g=h6,hh=h7;
    #pragma unroll 8
    for (int i = 0; i < 80; i++) {
        if (i >= 16)
            W[i&15] = EN_G1(W[(i-2)&15]) + EN_G0(W[(i-15)&15]) + W[(i-7)&15] + W[i&15];
        uint64_t t1 = hh + EN_S1(e) + EN_CH(e,f,g) + EN_SHA512_K[i] + W[i&15];
        uint64_t t2 = EN_S0(a) + EN_MAJ(a,b,c);
        hh=g;g=f;f=e;e=d_+t1;d_=c;c=b;b=a;a=t1+t2;
    }
    h0+=a;h1+=b;h2+=c;h3+=d_;h4+=e;h5+=f;h6+=g;h7+=hh;
    ec_store_be64(out,    h0); ec_store_be64(out+8,  h1);
    ec_store_be64(out+16, h2); ec_store_be64(out+24, h3);
    ec_store_be64(out+32, h4); ec_store_be64(out+40, h5);
    ec_store_be64(out+48, h6); ec_store_be64(out+56, h7);
}

__device__ __noinline__ void en_hmac_sha512(
        const uint8_t *key, uint32_t klen,
        const uint8_t *msg, uint32_t mlen,
        uint8_t out[64])
{
    uint8_t k[128]; memset(k,0,128);
    if(klen>128){ en_sha512_ctx t; en_sha512_init(&t); en_sha512_update(&t,key,klen); en_sha512_final(&t,k); }
    else memcpy(k,key,klen);
    uint8_t ipad[128],opad[128];
    for(int i=0;i<128;i++){ ipad[i]=k[i]^0x36; opad[i]=k[i]^0x5c; }
    en_sha512_ctx t;
    uint8_t inner[64];
    en_sha512_init(&t); en_sha512_update(&t,ipad,128); en_sha512_update(&t,msg,mlen); en_sha512_final(&t,inner);
    en_sha512_init(&t); en_sha512_update(&t,opad,128); en_sha512_update(&t,inner,64); en_sha512_final(&t,out);
}

__device__ __noinline__ void en_pbkdf2_hmac_sha512(
        const uint8_t *pw, uint32_t pwlen, uint8_t dk[64])
{
    /* Build key block: mnemonic padded/hashed to 128 bytes */
    uint8_t k[128]; memset(k, 0, 128);
    if (pwlen > 128) {
        en_sha512_ctx t; en_sha512_init(&t);
        en_sha512_update(&t, pw, pwlen); en_sha512_final(&t, k);
    } else {
        memcpy(k, pw, pwlen);
    }

    /* Compute ipad/opad mid-states: compress the 128-byte pad blocks once.
     * Store only the 8-word state (64 B), not the full ctx (200 B).
     * This eliminates 664 B of Local Memory spills in the hot loop. */
    uint64_t ipad_st[8], opad_st[8];
    {
        uint8_t pad[128];
        for (int i = 0; i < 128; i++) pad[i] = k[i] ^ 0x36;
        uint64_t h[8] = {
            0x6a09e667f3bcc908ULL,0xbb67ae8584caa73bULL,
            0x3c6ef372fe94f82bULL,0xa54ff53a5f1d36f1ULL,
            0x510e527fade682d1ULL,0x9b05688c2b3e6c1fULL,
            0x1f83d9abfb41bd6bULL,0x5be0cd19137e2179ULL };
        en_sha512_compress(h, pad);
        for (int i = 0; i < 8; i++) ipad_st[i] = h[i];

        for (int i = 0; i < 128; i++) pad[i] = k[i] ^ 0x5c;
        h[0]=0x6a09e667f3bcc908ULL;h[1]=0xbb67ae8584caa73bULL;
        h[2]=0x3c6ef372fe94f82bULL;h[3]=0xa54ff53a5f1d36f1ULL;
        h[4]=0x510e527fade682d1ULL;h[5]=0x9b05688c2b3e6c1fULL;
        h[6]=0x1f83d9abfb41bd6bULL;h[7]=0x5be0cd19137e2179ULL;
        en_sha512_compress(h, pad);
        for (int i = 0; i < 8; i++) opad_st[i] = h[i];
    }

    /* First HMAC: msg = "mnemonic\0\0\0\1" (12 bytes).
     * Inner: sha512(ipad_st || salt[12]) — total 140 bytes = 1120 bits.
     * Pad block: [salt[0..11]][0x80][zeros...][0x0000000000000460] */
    uint8_t U[64], T[64], inner[64];
    {
        uint64_t h[8];
        for (int i = 0; i < 8; i++) h[i] = ipad_st[i];
        uint8_t blk[128];
        const uint8_t sb[12] = {'m','n','e','m','o','n','i','c',0,0,0,1};
        for (int i = 0; i < 12; i++) blk[i] = sb[i];
        blk[12] = 0x80;
        for (int i = 13; i < 126; i++) blk[i] = 0;
        blk[126] = 0x04; blk[127] = 0x60;   /* 1120 bits big-endian */
        en_sha512_compress(h, blk);
        for (int i = 0; i < 8; i++) ec_store_be64(inner + i*8, h[i]);
    }
    /* Outer: sha512(opad_st || inner[64]) via sha512_resume_64 */
    sha512_resume_64(opad_st, inner, U);
    memcpy(T, U, 64);

    /* Hot loop: 2047 iterations, 2 compressions each (no ctx copies) */
    for (int i = 1; i < 2048; i++) {
        sha512_resume_64(ipad_st, U,     inner);
        sha512_resume_64(opad_st, inner, U);
        for (int j = 0; j < 64; j++) T[j] ^= U[j];
    }
    memcpy(dk, T, 64);
}

/* ================================================================
 * 256-bit arithmetic
 * ================================================================ */
typedef struct { uint32_t d[8]; } en_u256;

__constant__ en_u256 EN_FP = {{ 0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,
                                 0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF }};
__constant__ en_u256 EN_FN = {{ 0xD0364141,0xBFD25E8C,0xAF48A03B,0xBAAEDCE6,
                                 0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF }};

__device__ void en_u256_from_be(en_u256 *r, const uint8_t b[32]){
    for(int i=0;i<8;i++){
        int o=28-i*4;
        r->d[i]=((uint32_t)b[o]<<24)|((uint32_t)b[o+1]<<16)|((uint32_t)b[o+2]<<8)|((uint32_t)b[o+3]);
    }
}
__device__ void en_u256_to_be(const en_u256 *a, uint8_t b[32]){
    for(int i=0;i<8;i++){
        uint32_t w=a->d[i]; int o=28-i*4;
        b[o]=(w>>24)&0xFF;b[o+1]=(w>>16)&0xFF;b[o+2]=(w>>8)&0xFF;b[o+3]=w&0xFF;
    }
}

__device__ __forceinline__ int en_u256_cmp(const en_u256 *a, const en_u256 *b){
    for(int i=7;i>=0;i--){
        if(a->d[i]<b->d[i]) return -1;
        if(a->d[i]>b->d[i]) return  1;
    }
    return 0;
}
__device__ __forceinline__ int en_u256_is_zero(const en_u256 *a){
    uint32_t r=0; for(int i=0;i<8;i++) r|=a->d[i]; return r==0;
}
__device__ __forceinline__ uint32_t en_u256_add(en_u256 *r, const en_u256 *a, const en_u256 *b){
    uint64_t c=0;
    for(int i=0;i<8;i++){ uint64_t s=(uint64_t)a->d[i]+b->d[i]+c; r->d[i]=(uint32_t)s; c=s>>32; }
    return (uint32_t)c;
}
__device__ __forceinline__ uint32_t en_u256_sub(en_u256 *r, const en_u256 *a, const en_u256 *b){
    int64_t bw=0;
    for(int i=0;i<8;i++){ int64_t d=(int64_t)a->d[i]-b->d[i]+bw; r->d[i]=(uint32_t)d; bw=d>>32; }
    return (uint32_t)(-bw);
}

__device__ __forceinline__ void en_fp_reduce_once(en_u256 *r){
    if(en_u256_cmp(r,&EN_FP)>=0){ en_u256 t; en_u256_sub(&t,r,&EN_FP); *r=t; }
}
__device__ void en_fp_add(en_u256 *r, const en_u256 *a, const en_u256 *b){
    if(en_u256_add(r,a,b) || en_u256_cmp(r,&EN_FP)>=0){ en_u256 t; en_u256_sub(&t,r,&EN_FP); *r=t; }
}
__device__ void en_fp_sub(en_u256 *r, const en_u256 *a, const en_u256 *b){
    if(en_u256_sub(r,a,b)){ en_u256 t; en_u256_add(&t,r,&EN_FP); *r=t; }
}
__device__ __forceinline__ void en_fp_sqr(en_u256 *r, const en_u256 *a);

__device__ __noinline__ void en_fp_mul(en_u256 *r, const en_u256 *a, const en_u256 *b){
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
    uint64_t s[10];
    memset(s,0,sizeof(s));
    for(int i=0;i<8;i++){
        s[i]   += T[i];
        s[i]   += T[i+8]*977;
        s[i+1] += T[i+8];
    }
    for(int i=0;i<9;i++){ s[i+1]+=s[i]>>32; s[i]&=0xFFFFFFFF; }
    uint64_t hi=s[8]+(s[9]<<32); s[8]=0; s[9]=0;
    s[0]+=hi*977; s[1]+=hi;
    for(int i=0;i<8;i++){ s[i+1]+=s[i]>>32; s[i]&=0xFFFFFFFF; }
    if(s[8]){ s[0]+=977; s[1]+=1; for(int i=0;i<8;i++){ s[i+1]+=s[i]>>32; s[i]&=0xFFFFFFFF; } }
    for(int i=0;i<8;i++) r->d[i]=(uint32_t)s[i];
    en_fp_reduce_once(r);
    en_fp_reduce_once(r);
}

__device__ __forceinline__ void en_fp_sqr(en_u256 *r, const en_u256 *a){ en_fp_mul(r,a,a); }

__device__ __noinline__ void en_fp_inv(en_u256 *r, const en_u256 *a){
    en_u256 exp = EN_FP;
    exp.d[0] -= 2;
    en_u256 base = *a;
    memset(r,0,sizeof(*r)); r->d[0]=1;
    for(int bit=255;bit>=0;bit--){
        en_fp_sqr(r,r);
        if((exp.d[bit>>5]>>(bit&31))&1) en_fp_mul(r,r,&base);
    }
}

__device__ void en_fn_add(en_u256 *r, const en_u256 *a, const en_u256 *b){
    if(en_u256_add(r,a,b) || en_u256_cmp(r,&EN_FN)>=0){ en_u256 t; en_u256_sub(&t,r,&EN_FN); *r=t; }
}

typedef struct { en_u256 X,Y,Z; } en_jpoint;
typedef struct { en_u256 x,y;   } en_apoint;

__device__ __forceinline__ int en_jp_is_inf(const en_jpoint *p){ return en_u256_is_zero(&p->Z); }

__device__ __noinline__ void en_jp_to_affine(en_apoint *r, const en_jpoint *p){
    en_u256 zi,zi2,zi3;
    en_fp_inv(&zi,&p->Z);
    en_fp_sqr(&zi2,&zi);
    en_fp_mul(&zi3,&zi2,&zi);
    en_fp_mul(&r->x,&p->X,&zi2);
    en_fp_mul(&r->y,&p->Y,&zi3);
}

__device__ __noinline__ void en_jp_double(en_jpoint *r, const en_jpoint *p){
    if(en_jp_is_inf(p)){*r=*p;return;}
    en_u256 A,B,C,D,E,F,t;
    en_fp_sqr(&A,&p->X);
    en_fp_sqr(&B,&p->Y);
    en_fp_sqr(&C,&B);
    en_fp_add(&t,&p->X,&B); en_fp_sqr(&D,&t); en_fp_sub(&D,&D,&A); en_fp_sub(&D,&D,&C); en_fp_add(&D,&D,&D);
    en_fp_add(&E,&A,&A); en_fp_add(&E,&E,&A);
    en_fp_sqr(&F,&E);
    en_fp_mul(&r->Z,&p->Y,&p->Z); en_fp_add(&r->Z,&r->Z,&r->Z);
    en_fp_add(&t,&D,&D); en_fp_sub(&r->X,&F,&t);
    en_fp_sub(&t,&D,&r->X); en_fp_mul(&r->Y,&E,&t);
    en_u256 eC; en_fp_add(&eC,&C,&C);en_fp_add(&eC,&eC,&eC);en_fp_add(&eC,&eC,&eC);
    en_fp_sub(&r->Y,&r->Y,&eC);
}

typedef struct { uint8_t x[32]; uint8_t y[32]; } en_apoint_bytes;
__constant__ en_apoint_bytes EN_GT[15] = {
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

__device__ __noinline__ void en_jp_madd(en_jpoint *r, const en_jpoint *p, const en_u256 *x2, const en_u256 *y2){
    if(en_jp_is_inf(p)){ r->X=*x2; r->Y=*y2; memset(&r->Z,0,sizeof(r->Z)); r->Z.d[0]=1; return; }
    en_u256 Z1Z1,U2,Z1_3,S2,H,HH,I,J,rv,V,t;
    en_fp_sqr(&Z1Z1,&p->Z);
    en_fp_mul(&U2,x2,&Z1Z1);
    en_fp_mul(&Z1_3,&p->Z,&Z1Z1);
    en_fp_mul(&S2,y2,&Z1_3);
    en_fp_sub(&H,&U2,&p->X);
    if(en_u256_is_zero(&H)){
        en_u256 negY1; en_fp_sub(&negY1,&EN_FP,&p->Y); en_fp_reduce_once(&negY1);
        if(en_u256_cmp(&S2,&negY1)==0){ memset(r,0,sizeof(*r)); return; }
        en_jp_double(r,p); return;
    }
    en_fp_sqr(&HH,&H);
    en_fp_add(&I,&HH,&HH); en_fp_add(&I,&I,&I);
    en_fp_mul(&J,&H,&I);
    en_fp_sub(&rv,&S2,&p->Y); en_fp_add(&rv,&rv,&rv);
    en_fp_mul(&V,&p->X,&I);
    en_fp_sqr(&r->X,&rv);
    en_fp_sub(&r->X,&r->X,&J); en_fp_sub(&r->X,&r->X,&V); en_fp_sub(&r->X,&r->X,&V);
    en_u256 Y1J; en_fp_mul(&Y1J,&p->Y,&J); /* save Y1*J before r->Y overwrites p->Y */
    en_fp_sub(&t,&V,&r->X); en_fp_mul(&r->Y,&rv,&t);
    en_fp_add(&Y1J,&Y1J,&Y1J); en_fp_sub(&r->Y,&r->Y,&Y1J);
    en_fp_mul(&r->Z,&p->Z,&H); en_fp_add(&r->Z,&r->Z,&r->Z);
}

__device__ __noinline__ void en_ec_mul_G(en_jpoint *r, const en_u256 *k){
    /* 4-bit fixed window using EN_GT[0..14] (1G..15G in constant memory).
     * 256-bit scalar → 64 nibbles (MSB first).
     * Doublings: 63*4 = 252 (vs 256).  Max madd: 64 (vs ~128). */
    memset(r, 0, sizeof(*r));  /* infinity: Z=0 */
    for (int i = 63; i >= 0; i--) {
        /* 4 doublings before each nibble (skip the very first to avoid
         * doubling an infinity point unnecessarily). */
        if (i < 63) {
            en_jp_double(r, r); en_jp_double(r, r);
            en_jp_double(r, r); en_jp_double(r, r);
        }
        /* Extract 4-bit nibble at bit position [4i .. 4i+3].
         * i*4 is always a multiple of 4, so the nibble never crosses a
         * word boundary (bi ∈ {0,4,8,12,16,20,24,28}). */
        int shift = i * 4;
        uint32_t w = (k->d[shift >> 5] >> (shift & 31)) & 0xF;
        if (w == 0) continue;
        en_u256 gx, gy;
        en_u256_from_be(&gx, EN_GT[w - 1].x);
        en_u256_from_be(&gy, EN_GT[w - 1].y);
        en_jp_madd(r, r, &gx, &gy);
    }
}

typedef struct { uint8_t key[32]; uint8_t cc[32]; } en_bip32key;

__device__ void en_bip32_master(en_bip32key *out, const uint8_t seed[64]){
    const uint8_t bseed[12]={'B','i','t','c','o','i','n',' ','s','e','e','d'};
    uint8_t I[64];
    en_hmac_sha512(bseed,12,seed,64,I);
    memcpy(out->key,I,32); memcpy(out->cc,I+32,32);
}

__device__ __noinline__ void en_priv_to_cpub(const uint8_t priv[32], uint8_t pub[33]){
    en_u256 k; en_u256_from_be(&k,priv);
    en_jpoint P; en_ec_mul_G(&P,&k);
    en_apoint A; en_jp_to_affine(&A,&P);
    pub[0]=(A.y.d[0]&1)?0x03:0x02;
    en_u256_to_be(&A.x,pub+1);
}

__device__ __noinline__ void en_priv_to_upub64(const uint8_t priv[32], uint8_t pub[64]){
    en_u256 k; en_u256_from_be(&k,priv);
    en_jpoint P; en_ec_mul_G(&P,&k);
    en_apoint A; en_jp_to_affine(&A,&P);
    en_u256_to_be(&A.x,pub);
    en_u256_to_be(&A.y,pub+32);
}

__device__ __noinline__ void en_bip32_child_hard(en_bip32key *out, const en_bip32key *par, uint32_t idx){
    uint8_t data[37]; data[0]=0x00;
    memcpy(data+1,par->key,32);
    data[33]=(idx>>24)&0xFF; data[34]=(idx>>16)&0xFF;
    data[35]=(idx>> 8)&0xFF; data[36]=(idx    )&0xFF;
    uint8_t I[64];
    en_hmac_sha512(par->cc,32,data,37,I);
    en_u256 IL,pk,child;
    en_u256_from_be(&IL,I); en_u256_from_be(&pk,par->key);
    en_fn_add(&child,&IL,&pk);
    en_u256_to_be(&child,out->key); memcpy(out->cc,I+32,32);
}

__device__ __noinline__ void en_bip32_child_norm(en_bip32key *out, const en_bip32key *par, uint32_t idx){
    uint8_t data[37];
    en_priv_to_cpub(par->key,data);
    data[33]=(idx>>24)&0xFF; data[34]=(idx>>16)&0xFF;
    data[35]=(idx>> 8)&0xFF; data[36]=(idx    )&0xFF;
    uint8_t I[64];
    en_hmac_sha512(par->cc,32,data,37,I);
    en_u256 IL,pk,child;
    en_u256_from_be(&IL,I); en_u256_from_be(&pk,par->key);
    en_fn_add(&child,&IL,&pk);
    en_u256_to_be(&child,out->key); memcpy(out->cc,I+32,32);
}

__device__ __noinline__ void en_bip44_tron(const uint8_t seed[64], uint8_t privkey[32]){
    en_bip32key m,k1,k2,k3,k4,k5;
    en_bip32_master(&m,seed);
    en_bip32_child_hard(&k1,&m, 0x80000000u+44);
    en_bip32_child_hard(&k2,&k1,0x80000000u+195);
    en_bip32_child_hard(&k3,&k2,0x80000000u+0);
    en_bip32_child_norm(&k4,&k3,0);
    en_bip32_child_norm(&k5,&k4,0);
    memcpy(privkey,k5.key,32);
}

/* ================================================================
 * Keccak-256
 * ================================================================ */
#define EN_KRATE 136
#define EN_NROUNDS 24

__constant__ uint64_t EN_KRC[24]={
    0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
    0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
    0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
    0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
    0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
    0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL,
};
__constant__ int EN_KRHO[25]={ 0, 1,62,28,27,36,44, 6,55,20, 3,10,43,25,39,41,45,15,21, 8,18, 2,61,56,14 };
__constant__ int EN_KPI[25] ={ 0,10,20, 5,15,16, 1,11,21, 6, 7,17, 2,12,22,23, 8,18, 3,13,14,24, 9,19, 4 };

__device__ __noinline__ void en_keccak_f1600(uint64_t A[25]){
    for(int r=0;r<EN_NROUNDS;r++){
        uint64_t C[5],D[5];
        for(int x=0;x<5;x++) C[x]=A[x]^A[x+5]^A[x+10]^A[x+15]^A[x+20];
        for(int x=0;x<5;x++) D[x]=C[(x+4)%5]^EN_ROTR64(C[(x+1)%5],63);
        for(int i=0;i<25;i++) A[i]^=D[i%5];
        uint64_t B[25];
        for(int i=0;i<25;i++){
            int rho=EN_KRHO[i];
            B[EN_KPI[i]]=rho? EN_ROTR64(A[i],64-rho) : A[i];
        }
        for(int i=0;i<25;i++) A[i]=B[i]^(~B[(i/5)*5+(i%5+1)%5]&B[(i/5)*5+(i%5+2)%5]);
        A[0]^=EN_KRC[r];
    }
}

__device__ __noinline__ void en_keccak256(const uint8_t *data, uint32_t len, uint8_t out[32]){
    uint64_t st[25]; memset(st,0,200);
    uint32_t off=0;
    while(len-off>=EN_KRATE){
        for(int i=0;i<EN_KRATE/8;i++){ uint64_t w; memcpy(&w,data+off+i*8,8); st[i]^=w; }
        en_keccak_f1600(st); off+=EN_KRATE;
    }
    uint8_t last[EN_KRATE]; memset(last,0,EN_KRATE);
    uint32_t rem=len-off;
    memcpy(last,data+off,rem);
    last[rem]=0x01;
    last[EN_KRATE-1]^=0x80;
    for(int i=0;i<EN_KRATE/8;i++){ uint64_t w; memcpy(&w,last+i*8,8); st[i]^=w; }
    en_keccak_f1600(st);
    memcpy(out,st,32);
}

/* ================================================================
 * SHA-256  (BIP39 checksum validation)
 * ================================================================ */
__constant__ uint32_t EN_SHA256_K[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
};

#define EN_ROTR32(x,n) (((x)>>(n))|((x)<<(32-(n))))
#define EN_SS0(x) (EN_ROTR32(x,2) ^EN_ROTR32(x,13)^EN_ROTR32(x,22))
#define EN_SS1(x) (EN_ROTR32(x,6) ^EN_ROTR32(x,11)^EN_ROTR32(x,25))
#define EN_GS0(x) (EN_ROTR32(x,7) ^EN_ROTR32(x,18)^((x)>> 3))
#define EN_GS1(x) (EN_ROTR32(x,17)^EN_ROTR32(x,19)^((x)>>10))

__device__ __noinline__ void en_sha256_16(const uint8_t in[16], uint8_t out[32]) {
    uint32_t W[64];
    for (int i = 0; i < 4; i++)
        W[i] = ((uint32_t)in[i*4]<<24)|((uint32_t)in[i*4+1]<<16)|((uint32_t)in[i*4+2]<<8)|in[i*4+3];
    W[4] = 0x80000000u;
    for (int i = 5; i < 14; i++) W[i] = 0;
    W[14] = 0; W[15] = 128;
    for (int i = 16; i < 64; i++)
        W[i] = EN_GS1(W[i-2]) + W[i-7] + EN_GS0(W[i-15]) + W[i-16];
    uint32_t a=0x6a09e667,b=0xbb67ae85,c=0x3c6ef372,d=0xa54ff53a;
    uint32_t e=0x510e527f,f=0x9b05688c,g=0x1f83d9ab,hh=0x5be0cd19;
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = hh + EN_SS1(e) + EN_CH(e,f,g) + EN_SHA256_K[i] + W[i];
        uint32_t T2 = EN_SS0(a) + EN_MAJ(a,b,c);
        hh=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
    }
    uint32_t h[8] = {
        0x6a09e667+a, 0xbb67ae85+b, 0x3c6ef372+c, 0xa54ff53a+d,
        0x510e527f+e, 0x9b05688c+f, 0x1f83d9ab+g, 0x5be0cd19+hh
    };
    for (int i = 0; i < 8; i++) ec_store_be32(out + i*4, h[i]);
}

/* ================================================================
 * MurmurHash3-128 for 20-byte input (TRON address)
 * Identical to bits-and-blooms/bloom/v3 baseHashes() output.
 * ================================================================ */
#define BL_C1 0x87c37b91114253d5ULL
#define BL_C2 0x4cf5ad432745937fULL

__device__ __forceinline__ uint64_t bl_rotl64(uint64_t x, int r){
    return (x << r) | (x >> (64 - r));
}
__device__ __forceinline__ uint64_t bl_fmix64(uint64_t k){
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33; return k;
}

/* Load 8 bytes as little-endian uint64 */
__device__ __forceinline__ uint64_t bl_le64(const uint8_t *p){
    uint64_t v; memcpy(&v, p, 8); return v;  /* GPU is LE */
}

/* Returns four 64-bit hash values matching Go sum256() for 20-byte data */
__device__ void bloom_hash20(const uint8_t *data,
    uint64_t *h1, uint64_t *h2, uint64_t *h3, uint64_t *h4)
{
    uint64_t a1 = 0, a2 = 0;

    /* One full 16-byte block */
    uint64_t k1 = bl_le64(data),  k2 = bl_le64(data + 8);
    k1 *= BL_C1; k1 = bl_rotl64(k1, 31); k1 *= BL_C2; a1 ^= k1;
    a1 = bl_rotl64(a1, 27); a1 += a2; a1 = a1 * 5 + 0x52dce729ULL;
    k2 *= BL_C2; k2 = bl_rotl64(k2, 33); k2 *= BL_C1; a2 ^= k2;
    a2 = bl_rotl64(a2, 31); a2 += a1; a2 = a2 * 5 + 0x38495ab5ULL;

    /* 4-byte tail (bytes 16-19) */
    uint64_t tk1 =  ((uint64_t)data[19] << 24) | ((uint64_t)data[18] << 16)
                  | ((uint64_t)data[17] <<  8) |  (uint64_t)data[16];

    /* sum128(pad_tail=false, length=20, tail[0..3]) */
    {
        uint64_t t1 = a1, t2 = a2, p1 = tk1;
        p1 *= BL_C1; p1 = bl_rotl64(p1, 31); p1 *= BL_C2; t1 ^= p1;
        t1 ^= 20ULL; t2 ^= 20ULL;
        t1 += t2; t2 += t1;
        t1 = bl_fmix64(t1); t2 = bl_fmix64(t2);
        t1 += t2; t2 += t1;
        *h1 = t1; *h2 = t2;
    }

    /* sum128(pad_tail=true, length=21, tail[0..3])
     * pad_tail case 5: virtual extra byte at position 4 → k1 ^= (1<<32)    */
    {
        uint64_t t1 = a1, t2 = a2, p1 = tk1 | ((uint64_t)1 << 32);
        p1 *= BL_C1; p1 = bl_rotl64(p1, 31); p1 *= BL_C2; t1 ^= p1;
        t1 ^= 21ULL; t2 ^= 21ULL;
        t1 += t2; t2 += t1;
        t1 = bl_fmix64(t1); t2 = bl_fmix64(t2);
        t1 += t2; t2 += t1;
        *h3 = t1; *h4 = t2;
    }
}

/* Test a 20-byte address against the bloom filter.
 * Returns 1 if (possibly) present, 0 if definitely absent. */
__device__ int bloom_test(const uint8_t *addr,
    const uint64_t *bits, uint64_t m, uint32_t k)
{
    uint64_t h1, h2, h3, h4;
    bloom_hash20(addr, &h1, &h2, &h3, &h4);
    uint64_t h[4] = {h1, h2, h3, h4};
    for (uint32_t i = 0; i < k; i++) {
        uint64_t ii  = (uint64_t)i;
        uint64_t loc = h[ii & 1] + ii * h[2 + (((ii + (ii & 1)) & 3) >> 1)];
        uint64_t bit = loc % m;
        if (!((bits[bit >> 6] >> (bit & 63)) & 1)) return 0;
    }
    return 1;
}

/* ================================================================
 * Kernel 1: BIP39 filter – lightweight, no TRON derivation.
 * 256 threads/block, small stack (~1 KB/thread).
 * Outputs a compact array of valid word-index sets + original indices.
 * ================================================================ */
__global__ void bip39_filter_kernel(
    int64_t        start_idx,
    int64_t        total,
    const int16_t *known_words,
    const int8_t  *unk_pos,
    int8_t         unk_count,
    int16_t       *out_wi,       /* [capacity * 12] compacted word indices */
    int64_t       *out_idx,      /* [capacity] compacted original indices  */
    int            capacity,
    int           *out_count)
{
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int64_t idx = start_idx + tid;

    /* Decode word indices from integer index */
    int16_t wi[12];
    for (int i = 0; i < 12; i++) wi[i] = known_words[i];
    int64_t rem = idx;
    for (int i = 0; i < unk_count; i++) {
        wi[(int)unk_pos[i]] = (int16_t)(rem % 2048);
        rem /= 2048;
    }

    /* Pack 132 bits → 16 bytes entropy */
    uint32_t bits = 0;
    int      bc   = 0;
    uint8_t  entropy[16];
    int      eb   = 0;
    for (int i = 0; i < 12; i++) {
        bits = (bits << 11) | ((uint32_t)wi[i] & 0x7FFu);
        bc += 11;
        while (bc >= 8) {
            bc -= 8;
            if (eb < 16) entropy[eb++] = (uint8_t)(bits >> bc);
        }
    }
    /* Extract stored checksum: remaining bc bits in low bits of `bits`.
     * For 12 words: bc=4 always. Must use uint8_t shift to avoid 32-bit
     * carry masking out the low nibble (e.g. bits=0x...C3 with bc=4 →
     * (uint32_t shift) gives 0xC3, but correct cs = bits & 0xF = 0x3). */
    uint8_t stored_cs = (uint8_t)(bits & 0xF);

    /* BIP39 checksum validation – early exit for ~93.8% of threads */
    uint8_t sha_out[32];
    en_sha256_16(entropy, sha_out);
    if ((sha_out[0] >> 4) != stored_cs) return;

    /* Atomically reserve output slot */
    int slot = atomicAdd(out_count, 1);
    if (slot >= capacity) return;

    /* Write compact word indices and original index */
    for (int i = 0; i < 12; i++) out_wi[slot * 12 + i] = wi[i];
    out_idx[slot] = idx;
}

/* ================================================================
 * Kernel 2: TRON derive – heavy, operates only on BIP39-valid mnemonics.
 * 32 threads/block, requires cudaLimitStackSize = 65536.
 * 100% thread utilisation – no warp divergence.
 *
 * When bloom_bits != NULL, only bloom-passing addresses are written
 * via atomicAdd(out_count).  When NULL, all results are written and
 * out_count is unused (caller sets *out_count = count before launch).
 * ================================================================ */
__global__ void tron_derive_kernel(
    const int16_t  *wi_buf,       /* [count * 12] from kernel 1 */
    const int64_t  *idx_buf,      /* [count]       from kernel 1 */
    int             count,
    uint8_t        *out_addrs,    /* [count * 20] output */
    int64_t        *out_indices,  /* [count]      output */
    int            *out_count,    /* atomicAdd counter (bloom mode) */
    const uint64_t *bloom_bits,   /* GPU bloom bitset (NULL = disabled) */
    uint64_t        bloom_m,
    uint32_t        bloom_k)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;

    /* Read word indices */
    int16_t wi[12];
    for (int i = 0; i < 12; i++) wi[i] = wi_buf[(int64_t)tid * 12 + i];

    /* Build mnemonic string */
    uint8_t mn[120];
    int     ml = 0;
    for (int i = 0; i < 12; i++) {
        if (i > 0) mn[ml++] = ' ';
        const uint8_t *w = BIP39_WORDS[(int)wi[i]];
        for (int j = 0; j < 8 && w[j]; j++) mn[ml++] = w[j];
    }

    /* Full TRON derivation pipeline */
    uint8_t seed[64];
    en_pbkdf2_hmac_sha512(mn, (uint32_t)ml, seed);

    uint8_t priv[32];
    en_bip44_tron(seed, priv);

    uint8_t pub[64];
    en_priv_to_upub64(priv, pub);

    uint8_t khash[32];
    en_keccak256(pub, 64, khash);

    const uint8_t *addr = khash + 12;  /* 20-byte address */

    if (bloom_bits) {
        /* Bloom mode: only write addresses that pass the filter */
        if (!bloom_test(addr, bloom_bits, bloom_m, bloom_k)) return;
        int slot = atomicAdd(out_count, 1);
        out_indices[slot] = idx_buf[tid];
        memcpy(out_addrs + (int64_t)slot * 20, addr, 20);
    } else {
        /* No bloom: write all results at thread's own slot */
        out_indices[tid] = idx_buf[tid];
        memcpy(out_addrs + (int64_t)tid * 20, addr, 20);
    }
}

/* ================================================================
 * Host-side persistent state – one slot per GPU device (up to MAX_DEVICES).
 * Each DeviceState holds GPU-memory pointers allocated on the corresponding
 * device.  All host functions call cudaSetDevice(device_id) first so that
 * CUDA runtime operations target the correct device.
 * Not thread-safe within a single slot; callers must serialise per-device.
 * ================================================================ */

#define MAX_DEVICES 8

typedef struct {
    /* Intermediate buffers (kernel 1 → kernel 2) */
    int16_t *d_wi;
    int64_t *d_valid_idx;
    int     *d_valid_count;
    /* Output buffers (kernel 2 → host) */
    uint8_t *d_addrs;
    int64_t *d_indices;
    int     *d_out_count;   /* bloom-filtered output count */
    int64_t  buf_capacity;
    /* Persistent Bloom filter */
    uint64_t *d_bloom;
    uint64_t  bloom_m;
    uint32_t  bloom_k;
} DeviceState;

/* Zero-initialised: all pointers NULL, all counts 0. */
static DeviceState g_dev[MAX_DEVICES];

/* Ensure persistent buffers are (re-)allocated for the given device/capacity */
static int ensure_buffers(int device_id, int64_t capacity)
{
    DeviceState *ds = &g_dev[device_id];
    if (capacity <= ds->buf_capacity) return 0;
    /* Free old buffers */
    if (ds->d_wi)          { cudaFree(ds->d_wi);          ds->d_wi          = NULL; }
    if (ds->d_valid_idx)   { cudaFree(ds->d_valid_idx);   ds->d_valid_idx   = NULL; }
    if (ds->d_valid_count) { cudaFree(ds->d_valid_count); ds->d_valid_count = NULL; }
    if (ds->d_addrs)       { cudaFree(ds->d_addrs);       ds->d_addrs       = NULL; }
    if (ds->d_indices)     { cudaFree(ds->d_indices);     ds->d_indices     = NULL; }
    if (ds->d_out_count)   { cudaFree(ds->d_out_count);   ds->d_out_count   = NULL; }
    ds->buf_capacity = 0;

    if (cudaMalloc(&ds->d_wi,          (size_t)capacity * 12 * sizeof(int16_t)) != cudaSuccess) goto fail;
    if (cudaMalloc(&ds->d_valid_idx,   (size_t)capacity * sizeof(int64_t))      != cudaSuccess) goto fail;
    if (cudaMalloc(&ds->d_valid_count, sizeof(int))                             != cudaSuccess) goto fail;
    if (cudaMalloc(&ds->d_addrs,       (size_t)capacity * 20)                   != cudaSuccess) goto fail;
    if (cudaMalloc(&ds->d_indices,     (size_t)capacity * sizeof(int64_t))      != cudaSuccess) goto fail;
    if (cudaMalloc(&ds->d_out_count,   sizeof(int))                             != cudaSuccess) goto fail;
    ds->buf_capacity = capacity;
    return 0;
fail:
    if (ds->d_wi)          { cudaFree(ds->d_wi);          ds->d_wi          = NULL; }
    if (ds->d_valid_idx)   { cudaFree(ds->d_valid_idx);   ds->d_valid_idx   = NULL; }
    if (ds->d_valid_count) { cudaFree(ds->d_valid_count); ds->d_valid_count = NULL; }
    if (ds->d_addrs)       { cudaFree(ds->d_addrs);       ds->d_addrs       = NULL; }
    if (ds->d_indices)     { cudaFree(ds->d_indices);     ds->d_indices     = NULL; }
    if (ds->d_out_count)   { cudaFree(ds->d_out_count);   ds->d_out_count   = NULL; }
    return -1;
}

/* ================================================================
 * Host functions
 * ================================================================ */
extern "C" {

int gpu_enumerate_compute(
    int            device_id,
    int64_t        start_idx,
    int64_t        end_idx,
    const int16_t *known_words,
    const int8_t  *unknown_pos,
    int8_t         unknown_count,
    uint8_t       *out_addrs,
    int64_t       *out_indices,
    int            capacity,
    int           *out_count)
{
    if (cudaSetDevice(device_id) != cudaSuccess) return -1;
    DeviceState *ds = &g_dev[device_id];

    int64_t total = end_idx - start_idx;
    if (total <= 0) { *out_count = 0; return 0; }

    /* Ensure persistent buffers are large enough.
     * Also set the device stack limit once here; both kernels share the same
     * 65536-byte limit – Kernel 1 over-allocates (uses only ~400 bytes) but
     * avoids the expensive per-call cudaDeviceSetLimit + implicit sync. */
    if (ensure_buffers(device_id, (int64_t)capacity) != 0) return -1;
    cudaDeviceSetLimit(cudaLimitStackSize, 65536);

    /* Per-call small allocations: template only (12*2 + unk*1 bytes, tiny) */
    int16_t *d_known = NULL;
    int8_t  *d_unk   = NULL;
    if (cudaMalloc(&d_known, 12 * sizeof(int16_t))               != cudaSuccess) goto err;
    if (cudaMalloc(&d_unk,   (int)unknown_count * sizeof(int8_t)) != cudaSuccess) goto err;

    cudaMemcpy(d_known, known_words, 12 * sizeof(int16_t),                cudaMemcpyHostToDevice);
    cudaMemcpy(d_unk,   unknown_pos, (int)unknown_count * sizeof(int8_t), cudaMemcpyHostToDevice);
    cudaMemset(ds->d_valid_count, 0, sizeof(int));

    /* --- Kernel 1: BIP39 filter (256 threads/block) --- */
    {
        int bs = 256, nb = (int)((total + bs - 1) / bs);
        bip39_filter_kernel<<<nb, bs>>>(
            start_idx, total,
            d_known, d_unk, unknown_count,
            ds->d_wi, ds->d_valid_idx, capacity, ds->d_valid_count);
        if (cudaDeviceSynchronize() != cudaSuccess) goto err;
    }

    cudaFree(d_known); d_known = NULL;
    cudaFree(d_unk);   d_unk   = NULL;

    /* Retrieve valid count */
    {
        int cnt = 0;
        cudaMemcpy(&cnt, ds->d_valid_count, sizeof(int), cudaMemcpyDeviceToHost);
        if (cnt > capacity) cnt = capacity;

        if (cnt > 0) {
            /* --- Kernel 2: TRON derivation (32 threads/block) --- */
            int out_cnt;
            if (ds->d_bloom) {
                /* Bloom mode: output only matching addresses */
                cudaMemset(ds->d_out_count, 0, sizeof(int));
                int bs = 32, nb = (cnt + bs - 1) / bs;
                tron_derive_kernel<<<nb, bs>>>(
                    ds->d_wi, ds->d_valid_idx, cnt,
                    ds->d_addrs, ds->d_indices, ds->d_out_count,
                    ds->d_bloom, ds->bloom_m, ds->bloom_k);
                if (cudaDeviceSynchronize() != cudaSuccess) return -1;
                cudaMemcpy(&out_cnt, ds->d_out_count, sizeof(int), cudaMemcpyDeviceToHost);
                if (out_cnt > capacity) out_cnt = capacity;
            } else {
                /* No bloom: write all cnt results */
                int bs = 32, nb = (cnt + bs - 1) / bs;
                tron_derive_kernel<<<nb, bs>>>(
                    ds->d_wi, ds->d_valid_idx, cnt,
                    ds->d_addrs, ds->d_indices, NULL,
                    NULL, 0, 0);
                if (cudaDeviceSynchronize() != cudaSuccess) return -1;
                out_cnt = cnt;
            }

            *out_count = out_cnt;
            if (out_cnt > 0) {
                cudaMemcpy(out_addrs,   ds->d_addrs,   (size_t)out_cnt * 20,              cudaMemcpyDeviceToHost);
                cudaMemcpy(out_indices, ds->d_indices, (size_t)out_cnt * sizeof(int64_t), cudaMemcpyDeviceToHost);
            }
        } else {
            *out_count = 0;
        }
    }

    return 0;
err:
    if (d_known) cudaFree(d_known);
    if (d_unk)   cudaFree(d_unk);
    return -1;
}

/* Upload bloom filter to GPU persistent memory for the specified device. */
int gpu_bloom_upload(int device_id, const uint64_t *words, uint64_t word_count, uint64_t m, uint32_t k)
{
    if (cudaSetDevice(device_id) != cudaSuccess) return -1;
    DeviceState *ds = &g_dev[device_id];
    if (ds->d_bloom) { cudaFree(ds->d_bloom); ds->d_bloom = NULL; }
    if (cudaMalloc(&ds->d_bloom, word_count * sizeof(uint64_t)) != cudaSuccess) return -1;
    if (cudaMemcpy(ds->d_bloom, words, word_count * sizeof(uint64_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(ds->d_bloom); ds->d_bloom = NULL; return -1;
    }
    ds->bloom_m = m;
    ds->bloom_k = k;
    return 0;
}

/* Release GPU bloom filter memory for the specified device */
void gpu_bloom_free(int device_id)
{
    if (cudaSetDevice(device_id) != cudaSuccess) return;
    DeviceState *ds = &g_dev[device_id];
    if (ds->d_bloom) { cudaFree(ds->d_bloom); ds->d_bloom = NULL; }
    ds->bloom_m = 0; ds->bloom_k = 0;
}

/* Kernel: test a single address against GPU bloom filter */
__global__ void bloom_test_kernel(const uint8_t *addr, int *result,
    const uint64_t *bits, uint64_t m, uint32_t k)
{
    *result = bloom_test(addr, bits, m, k);
}

/* Test a 20-byte address against the bloom filter on the specified device.
 * Returns 1 if present, 0 if absent, -1 if no filter loaded. */
int gpu_bloom_test_addr(int device_id, const uint8_t *addr20)
{
    if (cudaSetDevice(device_id) != cudaSuccess) return -1;
    DeviceState *ds = &g_dev[device_id];
    if (!ds->d_bloom) return -1;

    uint8_t *d_addr = NULL;
    int     *d_result = NULL;
    int      h_result = 0;

    if (cudaMalloc(&d_addr,   20)          != cudaSuccess) return -1;
    if (cudaMalloc(&d_result, sizeof(int)) != cudaSuccess) {
        cudaFree(d_addr); return -1;
    }
    cudaMemcpy(d_addr, addr20, 20, cudaMemcpyHostToDevice);
    cudaMemset(d_result, 0, sizeof(int));

    bloom_test_kernel<<<1, 1>>>(d_addr, d_result, ds->d_bloom, ds->bloom_m, ds->bloom_k);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_addr);
    cudaFree(d_result);
    return h_result;
}

/* Debug kernel: run BIP39 checksum check on a single word set and return
 * stored_cs and sha_out[0] so the host can inspect what the GPU computed. */
__global__ void bip39_debug_kernel(const int16_t *wi_in, uint8_t *out_stored, uint8_t *out_sha)
{
    int16_t wi[12];
    for (int i = 0; i < 12; i++) wi[i] = wi_in[i];

    uint32_t bits = 0;
    int      bc   = 0;
    uint8_t  entropy[16];
    int      eb   = 0;
    for (int i = 0; i < 12; i++) {
        bits = (bits << 11) | ((uint32_t)wi[i] & 0x7FFu);
        bc += 11;
        while (bc >= 8) {
            bc -= 8;
            if (eb < 16) entropy[eb++] = (uint8_t)(bits >> bc);
        }
    }
    /* Same checksum extraction as bip39_filter_kernel */
    *out_stored = (uint8_t)(bits & 0xF);

    uint8_t sha_out[32];
    en_sha256_16(entropy, sha_out);
    *out_sha = sha_out[0];

    /* Also write entropy bytes for inspection */
    /* (re-using out_sha as entropy[0] is enough; caller checks sha[0]>>4) */
}

int gpu_bip39_debug(int device_id, const int16_t wi[12], uint8_t *stored_cs_out, uint8_t *sha0_out)
{
    if (cudaSetDevice(device_id) != cudaSuccess) return -1;
    int16_t *d_wi = NULL;
    uint8_t *d_stored = NULL, *d_sha = NULL;
    if (cudaMalloc(&d_wi,     12 * sizeof(int16_t)) != cudaSuccess) return -1;
    if (cudaMalloc(&d_stored, 1)                    != cudaSuccess) { cudaFree(d_wi); return -1; }
    if (cudaMalloc(&d_sha,    1)                    != cudaSuccess) { cudaFree(d_wi); cudaFree(d_stored); return -1; }

    cudaMemcpy(d_wi, wi, 12 * sizeof(int16_t), cudaMemcpyHostToDevice);
    bip39_debug_kernel<<<1, 1>>>(d_wi, d_stored, d_sha);
    cudaDeviceSynchronize();
    cudaMemcpy(stored_cs_out, d_stored, 1, cudaMemcpyDeviceToHost);
    cudaMemcpy(sha0_out,      d_sha,    1, cudaMemcpyDeviceToHost);

    cudaFree(d_wi); cudaFree(d_stored); cudaFree(d_sha);
    return 0;
}

/* Release all persistent GPU buffers for the specified device (call on shutdown) */
void gpu_enumerate_cleanup(int device_id)
{
    if (cudaSetDevice(device_id) != cudaSuccess) return;
    gpu_bloom_free(device_id);
    DeviceState *ds = &g_dev[device_id];
    if (ds->d_wi)          { cudaFree(ds->d_wi);          ds->d_wi          = NULL; }
    if (ds->d_valid_idx)   { cudaFree(ds->d_valid_idx);   ds->d_valid_idx   = NULL; }
    if (ds->d_valid_count) { cudaFree(ds->d_valid_count); ds->d_valid_count = NULL; }
    if (ds->d_addrs)       { cudaFree(ds->d_addrs);       ds->d_addrs       = NULL; }
    if (ds->d_indices)     { cudaFree(ds->d_indices);     ds->d_indices     = NULL; }
    if (ds->d_out_count)   { cudaFree(ds->d_out_count);   ds->d_out_count   = NULL; }
    ds->buf_capacity = 0;
}


} /* extern "C" */
