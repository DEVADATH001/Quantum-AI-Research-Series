OPENQASM 2.0;
include "qelib1.inc";
gate gate_PauliEvolution(param0) q0,q1 { ry(12.131374683759617) q0; }
gate gate_PauliEvolution_2877389536016(param0) q0,q1 { ry(6.718180499421129) q1; }
gate gate_PauliEvolution_2877389536816(param0) q0,q1 { sx q0; h q1; cx q1,q0; rz(0.005173442680467573) q0; cx q1,q0; sxdg q0; h q1; h q0; sx q1; cx q1,q0; rz(-0.005173442680467573) q0; cx q1,q0; h q0; sxdg q1; }
gate gate_EvolvedOps(param0,param1,param2) q0,q1 { x q0; gate_PauliEvolution(6.06568734187981) q0,q1; gate_PauliEvolution_2877389536016(-3.3590902497105657) q0,q1; gate_PauliEvolution_2877389536816(-0.005173442680467575) q0,q1; }
qreg q[2];
gate_EvolvedOps(6.06568734187981,-3.3590902497105657,-0.005173442680467575) q[0],q[1];