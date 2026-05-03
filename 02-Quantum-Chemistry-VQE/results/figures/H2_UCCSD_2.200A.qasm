OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(-12.995849759071112) q0; }
gate gate_PauliEvolution_2877353652976(param0) q0,q1 { ry(-5.853293177967417) q1; }
gate gate_PauliEvolution_2877353648816(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(0.004252873277139385) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-0.004252873277139385) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(-6.4979248795355575) q0,q1; gate_PauliEvolution_2877353652976(2.926646588983709) q0,q1; gate_PauliEvolution_2877353648816(-0.004252873277139387) q0,q1; }
qreg q[2];
gate_EvolvedOps(-6.4979248795355575,2.926646588983709,-0.004252873277139387) q[0],q[1];