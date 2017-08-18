Point(1) = {0.0, 0.0, 0, 1.0};
Point(2) = {15.0, 0.0, 0, 1.0};
Point(3) = {15.05, 0.0, 0, 1.0};
Point(4) = {50.0, 0.0, 0, 1.0};
Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Physical Line(100) = {1};   // mat=mat_1
Physical Line(200) = {2};   // mat=mat_3
Physical Line(300) = {3};   // mat=mat_2
Physical Point(10) = {1};  // bc=bc1
Physical Point(20) = {4};  // bc=bc2
Transfinite Line {1} = 6 Using Progression 1;
Transfinite Line {2} = 1 Using Progression 1;
Transfinite Line {3} = 15 Using Progression 1;
