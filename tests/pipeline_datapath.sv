// 3-Stage Pipelined Datapath with Self-Stimulus
// Stresses the simulator with pipeline registers, forwarding, multiple
// case statements, register file, and LFSR-generated instruction stream.
module pipeline_datapath (
    input  logic        clk,
    input  logic        rst,
    output logic [31:0] checksum,
    output logic [31:0] last_result
);

    // ── LFSR Instruction Generator ──────────────────────────────────────────
    logic [31:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 32'hCAFEBABE;
        else
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    // Decode fields from LFSR "instruction word"
    logic [2:0]  dec_op;
    logic [2:0]  dec_rd;
    logic [2:0]  dec_rs1;
    logic [2:0]  dec_rs2;
    logic [15:0] dec_imm16;

    assign dec_op   = lfsr[2:0];
    assign dec_rd   = lfsr[5:3];
    assign dec_rs1  = lfsr[8:6];
    assign dec_rs2  = lfsr[11:9];
    assign dec_imm16 = lfsr[27:12];

    // ── Register File (8 x 32-bit) ─────────────────────────────────────────
    logic [31:0] regfile [0:7];
    logic [31:0] rf_rd1;
    logic [31:0] rf_rd2;

    assign rf_rd1 = regfile[dec_rs1];
    assign rf_rd2 = regfile[dec_rs2];

    // ── Stage 1 → Stage 2 Pipeline Registers ───────────────────────────────
    logic [2:0]  s2_op;
    logic [2:0]  s2_rd;
    logic [31:0] s2_src1;
    logic [31:0] s2_src2;
    logic [31:0] s2_imm;

    always_ff @(posedge clk) begin
        if (rst) begin
            s2_op   <= 3'd0;
            s2_rd   <= 3'd0;
            s2_src1 <= 32'd0;
            s2_src2 <= 32'd0;
            s2_imm  <= 32'd0;
        end else begin
            s2_op   <= dec_op;
            s2_rd   <= dec_rd;
            s2_src1 <= rf_rd1;
            s2_src2 <= rf_rd2;
            s2_imm  <= {16'd0, dec_imm16};
        end
    end

    // ── Stage 2: ALU Execute ────────────────────────────────────────────────
    logic [31:0] alu_result;

    always_comb begin
        case (s2_op)
            3'd0: alu_result = s2_src1 + s2_src2;
            3'd1: alu_result = s2_src1 - s2_src2;
            3'd2: alu_result = s2_src1 & s2_src2;
            3'd3: alu_result = s2_src1 | s2_src2;
            3'd4: alu_result = s2_src1 ^ s2_src2;
            3'd5: alu_result = s2_src1 << s2_src2[4:0];
            3'd6: alu_result = s2_src1 >> s2_src2[4:0];
            3'd7: alu_result = s2_imm;
            default: alu_result = 32'd0;
        endcase
    end

    // ── Stage 2 → Stage 3 Pipeline Registers ───────────────────────────────
    logic [2:0]  s3_rd;
    logic [31:0] s3_result;

    always_ff @(posedge clk) begin
        if (rst) begin
            s3_rd     <= 3'd0;
            s3_result <= 32'd0;
        end else begin
            s3_rd     <= s2_rd;
            s3_result <= alu_result;
        end
    end

    // ── Stage 3: Writeback ──────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < 8; i = i + 1)
                regfile[i] <= 32'd0;
            checksum    <= 32'd0;
            last_result <= 32'd0;
        end else begin
            regfile[s3_rd] <= s3_result;
            checksum       <= checksum ^ s3_result;
            last_result    <= s3_result;
        end
    end

endmodule
