%{
#define SWIG_FILE_WITH_INIT
%}
%init %{
import_array();
%}
%include "numpy.i"
%apply (float* IN_ARRAY1, int DIM1) {(float* vals, int size)};
%apply (float* ARGOUT_ARRAY1, int DIM1) {(float *outvals, int size)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double *inputs, int size)};

%newobject *::copy;

%module mazesim
%{
#include "maze.h"
%}

class Environment
{
public:
    Environment(const Environment &e);
    void get_range(float &minx,float &miny, float &maxx, float& maxy);

    //initialize environment from maze file
    Environment(const char* filename);
    void display();
    float distance_to_start();
    float distance_to_target();
    float distance_to_poi();
    void generate_neural_inputs_wrapper(double* inputs, int size);
    void generate_neural_inputs(double* inputs);
    void interpret_outputs(float o1,float o2);
    void Update();
    bool collide_lines(Point loc,float rad);
    void update_rangefinders(Character& h);
    void update_radar(Character& h);
    int get_line_count();
    Line get_line(int idx);

    void update_radar_gen(Character& h,Point target, vector<float>& radar_arr);

    ~Environment();

    double closest_to_target;
    int steps;
    double closest_to_poi;
    vector<Line*> lines; //maze line segments
    Character hero; //navigator
    Point end; //the goal
    Point poi; //point of interest
    int reachpoi;
    int reachgoal;
    int get_sensor_size();
    bool goalattract;

};


class Character
{
public:
    vector<float> rangeFinderAngles; //angles of range finder sensors
    vector<float> radarAngles1; //beginning angles for radar sensors
    vector<float> radarAngles2; //ending angles for radar sensors

    vector<float> radar; //stores radar outputs
    vector<float> poi_radar; //stores poi radar
    vector<float> rangeFinders; //stores rangefinder outputs
    Point location;
        Point start;
    bool collide;
    int collisions;
    float total_spin;
    float heading;
    float speed;
    float ang_vel;
    float radius;
    float rangefinder_range;

	float total_dist;        
    Character();



};


class Point
{
public:
    Point(float x1,float y1)
    {
        x=x1;
        y=y1;
    }

    Point()
    {
    }

    Point(const Point& k)
    {
        x=k.x;
        y=k.y;
    }

    void fromfile(ifstream& file)
    {
        file >> x;
        file >> y;
    }

    //determine angle of vector defined by (0,0)->This Point
    float angle()
    {
        if(x==0.0)
        {
            if(y>0.0) return 90.0;
            return 270.0;
        }
        float ang=atan(y/x)/3.1415926*180.0;

        if(isnan(ang))
            cout << "NAN in angle\n";
        //quadrant 1 or 4
        if(x>0.0)
        {
            return ang;
        }
        return ang+180.0;
    }

    //rotate this point around another point
    void rotate(float angle,Point p)
    {
        float rad=angle/180.0*3.1415926;
        x-=p.x;
        y-=p.y;

        float ox=x;
        float oy=y;
        x=cos(rad)*ox-sin(rad)*oy;
        y=sin(rad)*ox+cos(rad)*oy;

        x+=p.x;
        y+=p.y;
    }
    //distance between this point and another point
    float distance(Point b)
    {
        float dx=b.x-x;
        float dy=b.y-y;
        return sqrt(dx*dx+dy*dy);
    }
    float x;
    float y;
};


class Line
{
public:
    Line(Point k,Point j)
    {
        a.x=k.x;
        a.y=k.y;
        b.x=j.x;
        b.y=j.y;
    }
    Line(ifstream& file)
    {
        a.fromfile(file);
        b.fromfile(file);
    }
    Line()
    {
    }
    //midpoint of the line segment
    Point midpoint()
    {
        Point newpoint;
        newpoint.x=(a.x+b.x)/2.0;
        newpoint.y=(a.y+b.y)/2.0;
        return newpoint;
    }

    //return point of intersection between two line segments if it exists
    Point intersection(Line L,bool &found)
    {

        Point pt(0.0,0.0);
        Point A(a);
        Point B(b);
        Point C(L.a);
        Point D(L.b);


        float rTop = (A.y-C.y)*(D.x-C.x)-(A.x-C.x)*(D.y-C.y);
        float rBot = (B.x-A.x)*(D.y-C.y)-(B.y-A.y)*(D.x-C.x);

        float sTop = (A.y-C.y)*(B.x-A.x)-(A.x-C.x)*(B.y-A.y);
        float sBot = (B.x-A.x)*(D.y-C.y)-(B.y-A.y)*(D.x-C.x);

        if ( (rBot == 0) || (sBot == 0))
        {
            //lines are parallel
            found = false;
            return pt;
        }

        float r = rTop/rBot;
        float s = sTop/sBot;

        if( (r > 0) && (r < 1) && (s > 0) && (s < 1) )
        {

            pt.x = A.x + r * (B.x - A.x);
            pt.y = A.y + r * (B.y - A.y);

            found=true;
            return pt;
        }

        else
        {

            found=false;
            return pt;
        }

    }

    //distance between line segment and point
    float distance(Point n)
    {
        float utop = (n.x-a.x)*(b.x-a.x)+(n.y-a.y)*(b.y-a.y);
        float ubot = a.distance(b);
        ubot*=ubot;
        if(ubot==0.0)
        {
            //cout << "Ubot zero?" << endl;
            return 0.0;
        }
        float u = utop/ubot;

        if(u<0 || u>1)
        {
            float d1=a.distance(n);
            float d2=b.distance(n);
            if(d1<d2) return d1;
            return d2;
        }
        Point p;
        p.x=a.x+u*(b.x-a.x);
        p.y=a.y+u*(b.y-a.y);
        return p.distance(n);
    }

    //line segment length
    float length()
    {
        return a.distance(b);
    }
    Point a;
    Point b;
};
