OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(6.714831075910166) q0; }
gate gate_PauliEvolution_2877389477744(param0) q0,q1 { ry(12.135297174391193) q1; }
gate gate_PauliEvolution_2877389486064(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(3.145981446991851) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-3.145981446991851) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(3.3574155379550845) q0,q1; gate_PauliEvolution_2877389477744(-6.067648587195598) q0,q1; gate_PauliEvolution_2877389486064(-3.145981446991852) q0,q1; }
qreg q[2];
gate_EvolvedOps(3.3574155379550845,-6.067648587195598,-3.145981446991852) q[0],q[1];