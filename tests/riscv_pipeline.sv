// 5-Stage RISC-V Pipeline with Self-Stimulus
// Stresses the simulator with pipeline registers, forwarding, register file,
// signed comparisons, sign-extension, and LFSR-generated instruction stream.
module riscv_pipeline (
    input  logic        clk,
    input  logic        rst,
    output logic [31:0] checksum,
    output logic [31:0] cycle_count
);

    // ── Opcode constants ──────────────────────────────────────────────────
    localparam logic [6:0] OP_RTYPE  = 7'b0110011;
    localparam logic [6:0] OP_IMM    = 7'b0010011;
    localparam logic [6:0] OP_LUI    = 7'b0110111;

    // ── Forward declarations for pipeline register outputs ───────────────
    logic [4:0]  mem_rd;
    logic [31:0] mem_result;
    logic [4:0]  wb_rd;
    logic [31:0] wb_result;

    // ── LFSR Instruction Generator (IF stage) ─────────────────────────────
    logic [31:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 32'hDEADC0DE;
        else
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    // ── Decode LFSR as instruction fields ─────────────────────────────────
    logic [6:0]  if_opcode;
    logic [4:0]  if_rd, if_rs1, if_rs2;
    logic [2:0]  if_funct3;
    logic [6:0]  if_funct7;
    logic [11:0] if_imm12;

    // Force opcodes into supported set using low bits
    logic [1:0] op_sel;
    assign op_sel = lfsr[1:0];
    assign if_opcode = (op_sel == 2'd0) ? OP_RTYPE :
                       (op_sel == 2'd1) ? OP_IMM :
                       (op_sel == 2'd2) ? OP_LUI : OP_RTYPE;
    assign if_rd     = lfsr[11:7];
    assign if_rs1    = lfsr[19:15];
    assign if_rs2    = lfsr[24:20];
    assign if_funct3 = lfsr[14:12];
    assign if_funct7 = lfsr[31:25];
    assign if_imm12  = lfsr[31:20];

    // ── IF/ID Pipeline Register ───────────────────────────────────────────
    logic [6:0]  id_opcode;
    logic [4:0]  id_rd, id_rs1, id_rs2;
    logic [2:0]  id_funct3;
    logic [6:0]  id_funct7;
    logic [31:0] id_imm;

    always_ff @(posedge clk) begin
        if (rst) begin
            id_opcode <= 7'd0;
            id_rd     <= 5'd0;
            id_rs1    <= 5'd0;
            id_rs2    <= 5'd0;
            id_funct3 <= 3'd0;
            id_funct7 <= 7'd0;
            id_imm    <= 32'd0;
        end else begin
            id_opcode <= if_opcode;
            id_rd     <= if_rd;
            id_rs1    <= if_rs1;
            id_rs2    <= if_rs2;
            id_funct3 <= if_funct3;
            id_funct7 <= if_funct7;
            // Sign-extend 12-bit immediate
            id_imm    <= {{20{if_imm12[11]}}, if_imm12};
        end
    end

    // ── Register File (32 x 32-bit, x0 hardwired to 0) ───────────────────
    logic [31:0] regfile [0:31];
    logic [31:0] rf_rs1_data, rf_rs2_data;

    assign rf_rs1_data = (id_rs1 == 5'd0) ? 32'd0 : regfile[id_rs1];
    assign rf_rs2_data = (id_rs2 == 5'd0) ? 32'd0 : regfile[id_rs2];

    // ── Data Forwarding (MEM→ID, WB→ID bypass) ──────────────────────────
    logic [31:0] fwd_src1, fwd_src2;

    assign fwd_src1 = (mem_rd != 5'd0 && mem_rd == id_rs1) ? mem_result :
                      (wb_rd  != 5'd0 && wb_rd  == id_rs1) ? wb_result  :
                      rf_rs1_data;
    assign fwd_src2 = (mem_rd != 5'd0 && mem_rd == id_rs2) ? mem_result :
                      (wb_rd  != 5'd0 && wb_rd  == id_rs2) ? wb_result  :
                      rf_rs2_data;

    // ── ID/EX Pipeline Register ───────────────────────────────────────────
    logic [6:0]  ex_opcode;
    logic [4:0]  ex_rd;
    logic [2:0]  ex_funct3;
    logic [6:0]  ex_funct7;
    logic [31:0] ex_rs1_data, ex_rs2_data, ex_imm;

    always_ff @(posedge clk) begin
        if (rst) begin
            ex_opcode   <= 7'd0;
            ex_rd       <= 5'd0;
            ex_funct3   <= 3'd0;
            ex_funct7   <= 7'd0;
            ex_rs1_data <= 32'd0;
            ex_rs2_data <= 32'd0;
            ex_imm      <= 32'd0;
        end else begin
            ex_opcode   <= id_opcode;
            ex_rd       <= id_rd;
            ex_funct3   <= id_funct3;
            ex_funct7   <= id_funct7;
            // Use forwarded values instead of raw register file reads
            ex_rs1_data <= fwd_src1;
            ex_rs2_data <= fwd_src2;
            ex_imm      <= id_imm;
        end
    end

    // ── EX Stage: ALU Execute (combinational) ─────────────────────────────
    logic [31:0] alu_result;
    logic [31:0] alu_a, alu_b;

    assign alu_a = ex_rs1_data;
    assign alu_b = (ex_opcode == OP_RTYPE) ? ex_rs2_data : ex_imm;

    always_comb begin
        case (ex_opcode)
            OP_LUI: alu_result = {ex_imm[31:12], 12'd0};
            OP_RTYPE, OP_IMM: begin
                case (ex_funct3)
                    3'd0: begin // ADD/SUB
                        if (ex_opcode == OP_RTYPE && ex_funct7[5])
                            alu_result = alu_a - alu_b;
                        else
                            alu_result = alu_a + alu_b;
                    end
                    3'd1: alu_result = alu_a << alu_b[4:0];        // SLL
                    3'd2: alu_result = ($signed(alu_a) < $signed(alu_b)) ? 32'd1 : 32'd0; // SLT
                    3'd3: alu_result = (alu_a < alu_b) ? 32'd1 : 32'd0; // SLTU
                    3'd4: alu_result = alu_a ^ alu_b;              // XOR
                    3'd5: begin // SRL/SRA
                        if (ex_opcode == OP_RTYPE && ex_funct7[5])
                            alu_result = $signed(alu_a) >>> alu_b[4:0]; // SRA
                        else
                            alu_result = alu_a >> alu_b[4:0];           // SRL
                    end
                    3'd6: alu_result = alu_a | alu_b;              // OR
                    3'd7: alu_result = alu_a & alu_b;              // AND
                    default: alu_result = 32'd0;
                endcase
            end
            default: alu_result = 32'd0;
        endcase
    end

    // ── EX/MEM Pipeline Register ──────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (rst) begin
            mem_rd     <= 5'd0;
            mem_result <= 32'd0;
        end else begin
            mem_rd     <= ex_rd;
            mem_result <= alu_result;
        end
    end

    // ── MEM/WB Pipeline Register ──────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (rst) begin
            wb_rd     <= 5'd0;
            wb_result <= 32'd0;
        end else begin
            wb_rd     <= mem_rd;
            wb_result <= mem_result;
        end
    end

    // ── Writeback Stage ───────────────────────────────────────────────────
    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < 32; i = i + 1)
                regfile[i] <= 32'd0;
            checksum    <= 32'd0;
            cycle_count <= 32'd0;
        end else begin
            if (wb_rd != 5'd0)
                regfile[wb_rd] <= wb_result;
            checksum    <= checksum ^ wb_result;
            cycle_count <= cycle_count + 32'd1;
        end
    end

endmodule
