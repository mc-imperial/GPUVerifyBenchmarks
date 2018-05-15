
typedef struct
{
    float4 m[3];
} matrice3x4;

typedef struct {
    float4 m[4];
} matrice4x4;

typedef struct{
	float3 A;	// origine
	float3 u;	// direction
} Rayon;

typedef struct Sphere{
	float3 C;	    // centre
	float  r;	    // rayon
	float  R,V,B,A;
} Sphere;

__constant__ matrice3x4 MView;  // matrice inverse de la matrice de vue

typedef struct Node {
	Sphere s;
	uint   fg, fd;
} Node;

#define numObj 4
__constant__ Node cnode[numObj];

static __attribute__((always_inline)) __device__ float intersectionSphere(Rayon R, float3 C, float r)
{
	float3 L(C-R.A);
	float d(dot(L,R.u)), l2(dot(L,L)), r2(r*r), m2, q, res;
  
	if( d < 0.0f && l2 > r2 ) {
		res = 0.0f;
	}
	else
	{
		m2 = l2 - d*d;
		if( m2 > r2 ) {
			res = 0.0f;
		}
		else
		{
			q = sqrt(r2-m2);
			if( l2 > r2 ) res = d - q;
			else res = d + q;
		}
	}
  
	return res;
}

static __attribute__((always_inline)) __device__ float intersectionPlan( Rayon R, float3 C, float3 N2 )
{
  float res;
  float3 N = normalize(make_float3(0.0f,1.0f,0.0f));
  float m(dot(N,R.u)), d, t;
  float3 L;
  
  if( fabs(m) < 0.0001f ) {
    res = 0.0f;
  }
  else {
    L = R.A - C;
    d = dot(N,L);
    t = -d/m;
    if( t > 0 ) {
      res = t;
    }
    else {
      res = 0.0f;
    }
  }
  
  return res;
}

static __attribute__((always_inline)) __device__ float3 getNormale(float3 P, float3 C)
{
  return normalize(P-C);
}

static __attribute__((always_inline)) __device__ float3 getNormaleP(float3 P)
{
  return normalize(make_float3(0.0f,1.0f,0.0f));
}

// multiplication d'un vecteur par une matrice (sans translation)
static __attribute__((always_inline)) __device__ float3 mul(matrice3x4 M, float3 v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// multiplication d'un vecteur par une matrice avec translation
static __attribute__((always_inline)) __device__ float4 mul(matrice3x4 M, float4 v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

static __attribute__((always_inline)) __device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp entre [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24)
		 | (uint(rgba.z*255)<<16)
		 | (uint(rgba.y*255)<<8 )
		 | (uint(rgba.x*255)    );
}

static __attribute__((always_inline)) __device__ bool notShadowRay( Node * node, float3 A, float3 u, float pas ) {
   float t(0.0f);
	Node  n;
	Rayon ray;
	float3 L(make_float3(10.0f,10.0f,10.0f)), tmp;
	float dst(dot(tmp=(L-A),tmp));
	ray.A = A+u*0.0001f;
	ray.u = u;
	for( int j(0); j < numObj && !t; j++ ) {
		n = cnode[j];
		n.s.C.x += pas;
		if( n.fg ){
         t = intersectionPlan(ray,n.s.C,n.s.C);
      }
		else{
         t = intersectionSphere(ray,n.s.C,n.s.r);
      }
		if( t > 0.0f && dot(tmp=(A+u*t),tmp) > dst ){
         t = 0.0f;
      }
	}
	return t == 0.0f;
}

static __attribute__((always_inline)) __device__ float float2int_pow20(float a)
{
   return a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a;
}

static __attribute__((always_inline)) __device__ float float2int_pow50(float a)
{
   return a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a* \
         a*a*a*a*a* a*a*a*a*a;

}
