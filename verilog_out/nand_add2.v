module nand_add2 (
    input  wire x0, x1, x2, x3,
    output wire y0, y1, y2
);

    wire l1_n0, l1_n1, l2_n0, l2_n1, l2_n2, l2_n6;
    wire l2_n8, l3_n0, l3_n1, l3_n2, l3_n3, l3_n5;
    wire l3_n6, l4_n0, l4_n1, l4_n2, l4_n3;

    // camada 1
    nand g_1_0 (l1_n0, x0, x2);
    nand g_1_1 (l1_n1, x1, x3);
    // camada 2
    nand g_2_0 (l2_n0, l1_n0, l1_n0);  // NOT(l1_n0)
    nand g_2_1 (l2_n1, l1_n1, l1_n1);  // NOT(l1_n1)
    nand g_2_2 (l2_n2, x3, x3);  // NOT(x3)
    nand g_2_6 (l2_n6, l1_n0, x0);
    nand g_2_8 (l2_n8, l1_n0, x2);
    // camada 3
    nand g_3_0 (l3_n0, l2_n0, l2_n8);
    nand g_3_1 (l3_n1, l2_n1, l2_n1);  // NOT(l2_n1)
    nand g_3_2 (l3_n2, l2_n2, l1_n0);
    nand g_3_3 (l3_n3, l2_n2, l2_n2);  // NOT(l2_n2)
    nand g_3_5 (l3_n5, l2_n1, l2_n0);
    nand g_3_6 (l3_n6, l2_n6, l2_n1);
    // camada 4
    nand g_4_0 (l4_n0, l3_n0, l3_n5);
    nand g_4_1 (l4_n1, l3_n1, l3_n6);
    nand g_4_2 (l4_n2, l3_n2, x1);
    nand g_4_3 (l4_n3, l3_n3, x1);
    // saida
    nand g_5_0 (y0, l4_n0, l4_n0);  // NOT(l4_n0)
    nand g_5_1 (y1, l4_n1, l4_n3);
    nand g_5_2 (y2, l4_n2, l4_n2);  // NOT(l4_n2)

endmodule

