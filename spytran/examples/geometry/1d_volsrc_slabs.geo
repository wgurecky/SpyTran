// Gmsh project created on Sun Nov 29 17:10:49 2015
Point(1) = {0.0, 0.0, 0, 1.0};
Point(2) = {4.0, 0.0, 0, 1.0};
Point(3) = {6.0, 0.0, 0, 1.0};
Point(4) = {9.0, 0.0, 0, 1.0};
Point(5) = {11.0, 0.0, 0, 1.0};
Point(6) = {14.0, 0.0, 0, 1.0};
Point(7) = {16.0, 0.0, 0, 1.0};
Point(8) = {20.0, 0.0, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Physical Line(101) = {1};   // mat=mat_1
Physical Line(102) = {2};   // mat=mat_2
Physical Line(103) = {3};   // mat=mat_1
Physical Line(104) = {4};   // mat=mat_2
Physical Line(105) = {5};   // mat=mat_1
Physical Line(106) = {6};   // mat=mat_2
Physical Line(107) = {7};   // mat=mat_1
Physical Point(10) = {1};   // bc=bc1
Physical Point(20) = {8};   // bc=bc2
Transfinite Line {1} = 5 Using Progression 1;
Transfinite Line {2} = 15 Using Progression 1;
Transfinite Line {3} = 5 Using Progression 1;
Transfinite Line {4} = 15 Using Progression 1;
Transfinite Line {5} = 5 Using Progression 1;
Transfinite Line {6} = 15 Using Progression 1;
Transfinite Line {7} = 5 Using Progression 1;
