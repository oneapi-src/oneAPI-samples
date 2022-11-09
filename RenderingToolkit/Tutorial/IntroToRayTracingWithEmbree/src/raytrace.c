/* raytrace.c
*
*    Example raytrace program
*/
#include <stdio.h>
#include <math.h>

typedef struct {
	double p[3];    /* starting point */
	double k[3];    /* direction of propagation */
	double n;       /* index of refraction */
	double q;       /* distance parameter */
	double gcosi;   /* incident k dot normal  */
	double gcosr;   /* refracted k dot normal  */
	double norm[3]; /* surface unit normal */
	int error;      /* error flag */
} RAY;

typedef struct {
	double cv;	/* curvature */
	double th;	/* axial thickness */
	double n;	/* index of refraction */
} SURF;

int nsurf = 5;

SURF surf[] = {
	{0.0, 1e20, 1.0},
	{0.0289603,9.0,1.67},
	{-0.0454959,2.5,1.728},
	{-0.0046592,43.55158,1.0},
	{0.0, 0.0, 1.0}
};

int raytrace(RAY *start, RAY data[]);
int trace(RAY *in, RAY *out, SURF *surf);
int print_ray(RAY *ray);
void print_vector(double v[3]);
double dot(double a[3], double b[3]);
void vnorm(double v[3], double norm);

int main()
{
	int i, iret;
	RAY ray[5], enp[1], *obj;
	obj = ray;
	/* define object point */
	obj->p[0] = 0.0;
	obj->p[1] = 0.0;
	obj->p[2] = 0.0;
	/* define entrance pupil aim point */
	enp->p[0] = 0.0;
	enp->p[1] = 12.5;
	enp->p[2] = 0.0;
	/* calculate ray optical direction cosines */
	for (i=0; i<3; i++) obj->k[i] = enp->p[i] - obj->p[i];
	obj->k[2] += surf->th;
	vnorm(obj->k,surf->n);
	for (i=0; i<3; i++) enp->k[i] = obj->k[i];
	enp->n = surf->n;
	printf("object point: ");
	print_vector(obj->p);
	raytrace(enp,ray);
	for (i=1; i<nsurf; i++) {
		printf("\nSurface %d\n",i);
		iret = print_ray(ray+i);
		if (iret<0) break;
	}
	return 0;
}

/*
*  Trace ray through optical system
*/
int raytrace(RAY *start, RAY data[])
{
	int k, image, iret;
	RAY *in, *out;
	SURF *sp;
	sp = surf+1;
	start->q = 0.0;
	in = start;
	out = data+1;
	image = nsurf-1;
	for (k=1; k<=image; k++) {
		iret = trace(in, out, sp++);
		if (iret<0) return iret;
		/* set pointers for next set of rays */
		in = out++;
	}
	return 0;
}

/*
*  Calculate surface intersection and direction of refraction
*/
int trace(RAY *in, RAY *out, SURF *surf)
{
	int i;
	double rni, rno, cv, q;
	double A,B,C,D;
	double root, power;
	rni = in->n;
	rno = surf->n;
	cv = surf->cv;
	/*
	* Transfer to coordinates of current surface.
	* on input, in->q should be axial distance to current surface
	* on output, in->q will be oblique distance to current surface
	*/
	out->p[0] = in->p[0];
	out->p[1] = in->p[1];
	out->p[2] = in->p[2] - in->q;
	/* intersect current surface */
	if (cv) {
		A = cv*rni*rni;
		B = in->k[2]-cv*dot(in->k,out->p);
		C = cv*dot(out->p,out->p)-2.0*out->p[2];
		D = B*B-A*C;
		if (D<0.0) {
			out->error = -1; /* missed surface */
			return out->error;
		}
		D = sqrt(D);
		q = C/(B+(B>0? D: -D));
	}
	else {
		if (in->k[2]==0.0) {
			out->error =  -1; /* ray parallel to plane */
			return out->error;
		}
		q = -out->p[2]/in->k[2];
	}
	in->q = q;
	out->q = surf->th;
	/* calculate point of intersection */
	for (i=0; i<3; i++) out->p[i] += q * in->k[i];
	/* calculate surface normal */
	for (i=0; i<3; i++) out->norm[i] = -cv*out->p[i];
	out->norm[2] += 1.0;
	/* refract ray into surface */
	out->gcosi = dot(in->k,out->norm);
	root = out->gcosi*out->gcosi + (rno+rni)*(rno - rni);
	if (root<0.0) {
		out->error = -2; /* total internal reflection */
		return out->error;
	}
	root = sqrt(root);
	out->gcosr = (out->gcosi>0.0? root: -root);
	power = out->gcosr - out->gcosi;
	for (i=0; i<3; i++) out->k[i] = in->k[i] + power*out->norm[i];
	out->n = rno;
	out->error = 0;
	return out->error;
}

/*
*  print ray data
*/
int print_ray(RAY *ray)
{
	if (ray->error==-1) {
		printf("MISSED SURFACE\n");
		return ray->error;
	}
	printf("surface intersection ");
	print_vector(ray->p);
	if (ray->error==0) {
		printf("optical direction cosines ");
		print_vector(ray->k);
	}
	printf("surface normal ");
	print_vector(ray->norm);
	printf("q %12.6f gcosi %12.6f gcosr %12.6f\n",
	       ray->q,ray->gcosi,ray->gcosr);
	if (ray->error==-2) {
		printf("TOTAL INTERNAL REFLECTION\n");
		return ray->error;
	}
	return ray->error;
}

/*
*  print vector data
*/
void print_vector(double v[3])
{
	printf("%12.6f %12.6f %12.6f\n",v[0],v[1],v[2]);
}

/*
*  vector dot product
*/
double dot(double a[3], double b[3])
{
	return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
}

/*
*  normalize vector length to specified value.
*/
void vnorm(double v[3], double norm)
{
	int i;
	double vn;
	vn = v[0]*v[0]+v[1]*v[1]+v[2]*v[2];
	if (vn==0.0) return;
	vn = norm/sqrt(vn);
	for (i=0; i<3; i++) v[i] *= vn;
}

