/* Geometry and forecast kernel used by the manuscript experiments.
   Compile with floating-point contraction disabled. */
#ifdef _WIN32
#define API __declspec(dllexport)
int _fltused=0;
#else
#define API __attribute__((visibility("default")))
#endif
typedef struct { double x, y; } Point;
typedef struct {
    int n;
    Point v[32];
    Point center, axes, unit, low, high;
} Obstacle;

static int inside(Point p, const Obstacle *o) {
    int odd=0;
    for (int i=0;i<o->n;++i) {
        Point a=o->v[i], b=o->v[(i+1)%o->n];
        double x=b.x-a.x, y=b.y-a.y, dx=p.x-a.x, dy=p.y-a.y;
        double cross=x*dy-y*dx;
        if (__builtin_fabs(cross)<=1e-10 && dx*x+dy*y>=-1e-10 &&
            (p.x-b.x)*x+(p.y-b.y)*y<=1e-10) return 0;
        if ((a.y>p.y)!=(b.y>p.y) && p.x<a.x+dy*x/y) odd=!odd;
    }
    return odd;
}

static int clear_segment(Point p,Point e,const Obstacle *o) {
    if ((p.x<o->low.x && e.x<o->low.x) || (p.x>o->high.x && e.x>o->high.x) ||
        (p.y<o->low.y && e.y<o->low.y) || (p.y>o->high.y && e.y>o->high.y)) return 1;
    double dx=e.x-p.x,dy=e.y-p.y;
    double px=p.x-o->center.x, py=p.y-o->center.y;
    double u=o->unit.x,v=o->unit.y;
    double lx=(px*u+py*v)/o->axes.x, ly=(-px*v+py*u)/o->axes.y;
    double sx=(dx*u+dy*v)/o->axes.x, sy=(-dx*v+dy*u)/o->axes.y;
    double aa=sx*sx+sy*sy;
    double t=aa>1e-25 ? -(lx*sx+ly*sy)/aa : 0;
    t=t<0?0:(t>1?1:t);
    double qx=lx+t*sx,qy=ly+t*sy;
    if (qx*qx+qy*qy<1-1e-12) return 0;
    /* A strict separating line excludes the full polygon. */
    int pos=1,neg=1;
    for (int i=0;i<o->n;++i) {
        Point a=o->v[i];
        double value=dx*(a.y-p.y)-dy*(a.x-p.x);
        pos &= value>1e-9; neg &= value< -1e-9;
    }
    if (pos||neg) return 1;
    if (inside(p,o)||inside(e,o)) return 0;
    double cuts[34];cuts[0]=0;cuts[1]=1;int nc=2;
    for (int i=0;i<o->n;++i) {
        Point a=o->v[i], b=o->v[(i+1)%o->n];
        double x=b.x-a.x, y=b.y-a.y, den=x*dy-y*dx;
        if (__builtin_fabs(den)<=1e-14) continue;
        t=-(x*(p.y-a.y)-y*(p.x-a.x))/den;
        if (t<=0 || t>=1) continue;
        int j=nc;
        while(j>0 && cuts[j-1]>t){cuts[j]=cuts[j-1];--j;}
        cuts[j]=t;++nc;
    }
    for (int i=1;i<nc;++i) {
        t=(cuts[i-1]+cuts[i])/2;
        Point q={p.x+t*dx,p.y+t*dy};
        if (inside(q,o)) return 0;
    }
    return 1;
}

static float sample(const float *sdf,int ny,double x,double y) {
    int i=(int)x,j=(int)y;
    double wx=1-(x-i),wy=1-(y-j);
    const float *z=sdf+i*ny+j;
    double result=((double)z[0]*wx)*wy;
    result+=((double)z[1]*wx)*(1-wy);
    result+=((double)z[ny]*(1-wx))*wy;
    result+=((double)z[ny+1]*(1-wx))*(1-wy);
    return (float)result;
}


API void segments(const Point *p,const Point *e,int n,const Obstacle *obs,
                                     int no,unsigned char *out) {
    for(int k=0;k<n;++k) {
        int vis=1;
        for(int j=0;j<no && vis;++j) vis=clear_segment(p[k],e[k],&obs[j]);
        out[k]=(unsigned char)vis;
    }
}

/* A separating projection of the convex hull of the four sweep endpoints
   proves that the entire moving sight segment misses the component. No
   separation found => defer to the ordinary continuous event routine. */
API int clear_sweep(const Point *quad,const Obstacle *obs,int no) {
    Point normals[8]={{1,0},{0,1}};int nn=2;
    for(int i=0;i<4;++i) for(int j=0;j<i;++j) {
        normals[nn].x=quad[i].y-quad[j].y;
        normals[nn++].y=quad[j].x-quad[i].x;
    }
    for(int j=0;j<no;++j) {
        const Obstacle *o=obs+j;int poly=0,ellipse=0;
        for(int k=0;k<nn && !(poly && ellipse);++k) {
            Point n=normals[k];
            double low=1e300,high=-1e300;
            for(int i=0;i<4;++i) {
                double z=n.x*quad[i].x+n.y*quad[i].y;
                if(z<low)low=z;if(z>high)high=z;
            }
            double eps=1e-9*(1+__builtin_fabs(n.x)+__builtin_fabs(n.y));
            low-=eps;high+=eps;
            if(!poly) {
                int below=1,above=1;
                for(int i=0;i<o->n;++i) {
                    double z=n.x*o->v[i].x+n.y*o->v[i].y;
                    below &= z<low;above &= z>high;
                }
                poly=below||above;
            }
            if(!ellipse) {
                double z=n.x*o->center.x+n.y*o->center.y;
                double gap=z<low?low-z:(z>high?z-high:0);
                double a=o->axes.x*(n.x*o->unit.x+n.y*o->unit.y);
                double b=o->axes.y*(-n.x*o->unit.y+n.y*o->unit.x);
                ellipse=gap*gap>a*a+b*b;
            }
        }
        if(!(poly && ellipse))return 0;
    }
    return 1;
}

API int forecast(const Point *pp,const Point *ee,int nt,int np,int ne,
    const Obstacle *obs,int no,const float *sdf,int nx,int ny,Point origin,double spacing,
    const double *fractions,int ns,unsigned char *visible,double *clearance) {
    for (int it=0;it<nt;++it) for (int i=0;i<np;++i) for (int j=0;j<ne;++j) {
        int k=(it*np+i)*ne+j;
        if (it==0 && k){ visible[k]=visible[0];clearance[k]=clearance[0];continue; }
        Point p=pp[it*np+i],e=ee[it*ne+j];
        int vis=1;
        for (int o=0;o<no && vis;++o) vis=clear_segment(p,e,&obs[o]);
        visible[k]=(unsigned char)vis;
        double dx=e.x-p.x,dy=e.y-p.y;
        double best=1e300;
        for(int s=0;s<ns;++s) {
            double x=(p.x+fractions[s]*dx-origin.x)/spacing;
            double y=(p.y+fractions[s]*dy-origin.y)/spacing;
            if (x<0 || y<0 || x>=nx-1 || y>=ny-1) return 1;
            float val=sample(sdf,ny,x,y);
            if (val<best) best=val;
        }
        clearance[k]=vis?(best>0?best:0):(best<0?best:0);
    }
    return 0;
}
