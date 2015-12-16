Mesh.Algorithm = 1;
cl = 0.04;
cl2 = 0.08;
Point(1) = {0, 0, 0, cl2};
Point(2) = {.665, 0, 0, cl};
Point(3) = {.665, .665, 0, cl};
Point(4) = {.0, .665, 0, cl};
Point(5) = {.412, .0, 0, cl};
Point(6) = {.0, .412, 0, cl};
Point(7) = {.475, .0, 0, cl};
Point(8) = {.0, .475, 0, cl};
Line(1) = {1, 5};
Line(2) = {5, 7};
Line(3) = {7, 2};
Line(4) = {2, 3};
Line(5) = {3, 4};
Line(6) = {4, 8};
Line(7) = {8, 6};
Line(8) = {6, 1};
Circle(9) = {5, 1, 6};
Circle(10) = {7, 1, 8};
Line Loop(11) = {8, 1, 9};
Ruled Surface(12) = {11};
Line Loop(13) = {9, -7, -10, -2};
Ruled Surface(14) = {13};
Line Loop(15) = {3, 4, 5, 6, -10};
Plane Surface(16) = {15};
Physical Line(17) = {1, 2, 3};    // bc=bc1
Physical Line(18) = {4};          // bc=bc2
Physical Line(19) = {5};          // bc=bc3
Physical Line(20) = {6, 7, 8};    // bc=bc4
Physical Surface(21) = {12};      // mat=mat_1
Physical Surface(22) = {14};      // mat=mat_2
Physical Surface(23) = {16};      // mat=mat_3
