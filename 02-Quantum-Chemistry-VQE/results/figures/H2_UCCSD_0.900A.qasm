OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(-3.5752754912961624) q0; }
gate gate_PauliEvolution_2113329470576(param0) q0,q1 { ry(9.860457514169957) q1; }
gate gate_PauliEvolution_2113329470736(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(4.6197204455138365) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-4.6197204455138365) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(-1.7876377456480819) q0,q1; gate_PauliEvolution_2113329470576(-4.93022875708498) q0,q1; gate_PauliEvolution_2113329470736(-4.619720445513838) q0,q1; }
qreg q[2];
gate_EvolvedOps(-1.7876377456480819,-4.93022875708498,-4.619720445513838) q[0],q[1];