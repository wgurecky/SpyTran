cl = 0.2;
Point(1) = {0, 0, 0, cl};
Point(2) = {0, 1, 0, cl};
Point(3) = {1, 1, 0, cl};
Point(4) = {1, 0, 0, cl};
Point(5) = {0.5, 0.5, 0, cl};
Point(6) = {0.9, 0.5, 0, cl};
Point(7) = {0.1, 0.5, 0, cl};
Point(8) = {0.5, 0.9, 0, cl};
Point(9) = {0.5, 0.1, 0, cl};
Line(1) = {1, 4};
Line(2) = {4, 3};
Line(3) = {3, 2};
Line(4) = {2, 1};
Circle(5) = {6, 5, 8};
Circle(6) = {8, 5, 7};
Circle(7) = {7, 5, 9};
Circle(8) = {9, 5, 6};
Line Loop(11) = {3, 4, 1, 2, -5, -8, -7, -6};
Plane Surface(11) = {11};
Line Loop(12) = {6, 7, 8, 5};
Ruled Surface(12) = {12};
Physical Line(13) = {4};      // bc=bc1
Physical Line(14) = {3};      // bc=bc1
Physical Line(15) = {2};      // bc=bc1
Physical Line(16) = {1};      // bc=bc1
Physical Surface(17) = {12};  // mat=mat_1
Physical Surface(18) = {11};  // mat=mat_2
