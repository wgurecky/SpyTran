// Gmsh project created on Sun Nov 29 17:10:49 2015
Point(1) = {0.0, 0.0, 0, 1.0};
Point(2) = {4.0, 0.0, 0, 1.0};
Point(3) = {6.0, 0.0, 0, 1.0};
Point(4) = {9.0, 0.0, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Physical Line(101) = {1};   // mat=mat_1
Physical Line(102) = {2};   // mat=mat_2
Physical Line(103) = {3};   // mat=mat_1
Physical Point(10) = {1};   // bc=bc1
Physical Point(20) = {4};   // bc=bc2
Transfinite Line {1} = 20 Using Progression 1;
Transfinite Line {2} = 40 Using Progression 1;
Transfinite Line {3} = 20 Using Progression 1;
