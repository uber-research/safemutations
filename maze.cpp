#include <vector>
#include <fstream>
#include <iostream>
using namespace std;
class Point
{
public:
    void fromfile(ifstream& file)
    {
        file >> x;
        file >> y;
    }
    int x;
    int y;
};
class Line
{
public:
    Line(ifstream& file)
    {
        a.fromfile(file);
        b.fromfile(file);
    }
    Line()
    {
    }
    Point a;
    Point b;
};
class Character
{
public:
    Character()
    {
        heading=0.0f;
        speed=0.0f;
        ang_vel=0.0f;
    }
    Point location;
    float heading;
    float speed;
    float ang_vel;
};
class Environment
{
public:
    Environment(const char* filename)
    {
        ifstream inpfile(filename);
        hero.location.fromfile(inpfile);
        end.fromfile(inpfile);
        while(!inpfile.eof())
        {
            Line* x=new Line(inpfile);
            lines.push_back(x);
        }
    }
    ~Environment()
    {
        //clean up lines!
    }
    vector<Line*> lines;
    Character hero;
    Point end;
};



