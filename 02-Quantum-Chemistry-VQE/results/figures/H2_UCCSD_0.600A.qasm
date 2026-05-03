OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(-3.575620123370905) q0; }
gate gate_PauliEvolution_2113329200848(param0) q0,q1 { ry(9.862204411983976) q1; }
gate gate_PauliEvolution_2113329201008(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(4.620981291986577) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-4.620981291986577) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(-1.787810061685453) q0,q1; gate_PauliEvolution_2113329200848(-4.93110220599199) q0,q1; gate_PauliEvolution_2113329201008(-4.6209812919865785) q0,q1; }
qreg q[2];
gate_EvolvedOps(-1.787810061685453,-4.93110220599199,-4.6209812919865785) q[0],q[1];