// Gmsh project created on Sun Nov 29 17:10:49 2015
Point(1) = {0.0, 0.0, 0, 1.0};
Point(2) = {1.0, 0.0, 0, 1.0};
Point(3) = {2.0, 0.0, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Physical Line(100) = {1};   // mat=mat_1
Physical Line(200) = {2};   // mat=mat_2
Physical Point(10) = {1};  // bc=bc1
Physical Point(20) = {3};  // bc=bc2
Transfinite Line {1} = 10 Using Progression 1;
Transfinite Line {2} = 10 Using Progression 1;
