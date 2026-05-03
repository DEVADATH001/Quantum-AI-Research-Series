OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(12.129979041002201) q0; }
gate gate_PauliEvolution_2877389868080(param0) q0,q1 { ry(6.719581357584962) q1; }
gate gate_PauliEvolution_2877389865040(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(0.004937721211542097) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-0.004937721211542097) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(6.064989520501102) q0,q1; gate_PauliEvolution_2877389868080(-3.3597906787924825) q0,q1; gate_PauliEvolution_2877389865040(-0.004937721211542099) q0,q1; }
qreg q[2];
gate_EvolvedOps(6.064989520501102,-3.3597906787924825,-0.004937721211542099) q[0],q[1];