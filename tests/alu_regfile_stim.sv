// Self-stimulus ALU + register file benchmark
// Embeds an LFSR to generate pseudo-random inputs every cycle,
// eliminating degenerate all-zero-input benchmarking.
module alu_regfile_stim (
    input  logic        clk,
    input  logic        rst,
    output logic [31:0] result
);

    // 32-bit Galois LFSR for self-stimulus (x^32 + x^22 + x^2 + x + 1)
    logic [31:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 32'hDEADBEEF;
        else
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    // Derive control signals from LFSR bits
    logic [2:0]  op;
    logic [2:0]  rd;
    logic [2:0]  rs1;
    logic [2:0]  rs2;
    logic [31:0] imm;

    assign op  = lfsr[2:0];
    assign rd  = lfsr[5:3];
    assign rs1 = lfsr[8:6];
    assign rs2 = lfsr[11:9];
    assign imm = lfsr;

    // Register file (8 x 32-bit)
    logic [31:0] regfile [0:7];
    logic [31:0] src1;
    logic [31:0] src2;

    assign src1 = regfile[rs1];
    assign src2 = regfile[rs2];

    // ALU + register write
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < 8; i = i + 1)
                regfile[i] <= 32'd0;
            result <= 32'd0;
        end else begin
            case (op)
                3'd0: result <= src1 + src2;
                3'd1: result <= src1 - src2;
                3'd2: result <= src1 & src2;
                3'd3: result <= src1 | src2;
                3'd4: result <= src1 ^ src2;
                3'd5: result <= src1 << src2[4:0];
                3'd6: result <= src1 >> src2[4:0];
                3'd7: result <= imm;
                default: result <= 32'd0;
            endcase
            regfile[rd] <= result;
        end
    end

endmodule
