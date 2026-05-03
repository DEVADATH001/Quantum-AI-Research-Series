OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(0.4286412828722271) q0; }
gate gate_PauliEvolution_2877354075920(param0) q0,q1 { ry(5.853602381502863) q1; }
gate gate_PauliEvolution_2877354076240(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(3.1454086298239092) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-3.1454086298239092) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(0.21432064143611362) q0,q1; gate_PauliEvolution_2877354075920(-2.926801190751432) q0,q1; gate_PauliEvolution_2877354076240(-3.14540862982391) q0,q1; }
qreg q[2];
gate_EvolvedOps(0.21432064143611362,-2.926801190751432,-3.14540862982391) q[0],q[1];