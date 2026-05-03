OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(2.710423406945957) q0; }
gate gate_PauliEvolution_2877389139120(param0) q0,q1 { ry(3.572448141038026) q1; }
gate gate_PauliEvolution_2877389143920(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(4.621011457231035) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-4.621011457231035) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(1.355211703472979) q0,q1; gate_PauliEvolution_2877389139120(-1.7862240705190138) q0,q1; gate_PauliEvolution_2877389143920(-4.621011457231036) q0,q1; }
qreg q[2];
gate_EvolvedOps(1.355211703472979,-1.7862240705190138,-4.621011457231036) q[0],q[1];