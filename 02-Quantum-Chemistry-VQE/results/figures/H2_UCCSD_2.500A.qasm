OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(3.572754912464136) q0; }
gate gate_PauliEvolution_2875605995024(param0) q0,q1 { ry(8.99673466449831) q1; }
gate gate_PauliEvolution_2875605995184(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(1.480771807123962) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-1.480771807123962) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(1.7863774562320687) q0,q1; gate_PauliEvolution_2875605995024(-4.498367332249157) q0,q1; gate_PauliEvolution_2875605995184(-1.4807718071239624) q0,q1; }
qreg q[2];
gate_EvolvedOps(1.7863774562320687,-4.498367332249157,-1.4807718071239624) q[0],q[1];