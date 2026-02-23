module alu_regfile (
    input  logic        clk,
    input  logic        rst,
    input  logic [2:0]  op,
    input  logic [2:0]  rd,
    input  logic [2:0]  rs1,
    input  logic [2:0]  rs2,
    input  logic [31:0] imm,
    output logic [31:0] result
);

    logic [31:0] regfile [0:7];
    logic [31:0] src1;
    logic [31:0] src2;

    // Combinational read
    assign src1 = regfile[rs1];
    assign src2 = regfile[rs2];

    // ALU + register write
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < 8; i = i + 1) begin
                regfile[i] <= 32'd0;
            end
            result <= 32'd0;
        end else begin
            case (op)
                3'd0: result <= src1 + src2;        // ADD
                3'd1: result <= src1 - src2;        // SUB
                3'd2: result <= src1 & src2;        // AND
                3'd3: result <= src1 | src2;        // OR
                3'd4: result <= src1 ^ src2;        // XOR
                3'd5: result <= src1 << src2[4:0];  // SHL
                3'd6: result <= src1 >> src2[4:0];  // SHR
                3'd7: result <= imm;                // LUI
                default: result <= 32'd0;
            endcase
            regfile[rd] <= result;
        end
    end

endmodule
