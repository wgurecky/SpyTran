// Gmsh project created on Sun Dec 13 19:38:57 2015
cl = 0.2;
cl2 = 0.1;
Point(1) = {0, 0, 0, cl};
Point(2) = {4, 0, 0, cl2};
Point(3) = {6, 0, 0, cl2};
Point(4) = {9, 0, 0, cl};
Point(5) = {0, 0.5, 0, cl};
Point(6) = {4, 0.5, 0, cl2};
Point(7) = {6, 0.5, 0, cl2};
Point(8) = {9, 0.5, 0, cl};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 8};
Line(5) = {8, 7};
Line(6) = {7, 6};
Line(7) = {6, 5};
Line(8) = {5, 1};
Line(9) = {2, 6};
Line(10) = {3, 7};
Line Loop(11) = {8, 1, 9, 7};
Plane Surface(12) = {11};
Line Loop(13) = {2, 10, 6, -9};
Plane Surface(14) = {13};
Line Loop(15) = {3, 4, 5, -10};
Plane Surface(16) = {15};
Physical Line(17) = {1, 2, 3};   // bc=bc1
Physical Line(18) = {4};         // bc=bc2
Physical Line(19) = {5, 6, 7};   // bc=bc3
Physical Line(20) = {8};         // bc=bc4
Physical Surface(21) = {12};     // mat=mat_1
Physical Surface(22) = {14};     // mat=mat_2
Physical Surface(23) = {16};     // mat=mat_1
