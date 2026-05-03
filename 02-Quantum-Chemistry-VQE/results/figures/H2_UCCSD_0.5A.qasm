OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(-2.702565273322704) q0; }
gate gate_PauliEvolution_2633650816176(param0) q0,q1 { ry(-3.5805769527539892) q1; }
gate gate_PauliEvolution_2633650818736(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(-4.8065429604463885) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(4.8065429604463885) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(-1.3512826366613524) q0,q1; gate_PauliEvolution_2633650816176(1.7902884763769953) q0,q1; gate_PauliEvolution_2633650818736(4.80654296044639) q0,q1; }
qreg q[2];
gate_EvolvedOps(-1.3512826366613524,1.7902884763769953,4.80654296044639) q[0],q[1];