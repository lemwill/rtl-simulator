// Barrel shifter with packed bit assignment and dynamic range select
module barrel_shifter (
    input  logic        clk,
    input  logic        rst,
    output logic [31:0] result
);
    // LFSR for stimulus
    logic [31:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 32'hDEADBEEF;
        else
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    // Extract fields from LFSR
    logic [4:0]  shift_amt;
    logic [1:0]  op;
    logic [31:0] data;
    assign shift_amt = lfsr[4:0];
    assign op        = lfsr[6:5];
    assign data      = lfsr;

    // Barrel shift operations
    logic [31:0] shifted;
    always_comb begin
        case (op)
            2'd0: shifted = data << shift_amt;         // SHL
            2'd1: shifted = data >> shift_amt;         // SHR
            2'd2: shifted = data >>> shift_amt;        // SRA (arithmetic)
            2'd3: shifted = {data, data} >> shift_amt; // rotate right
            default: shifted = data;
        endcase
    end

    // Dynamic range select: extract a byte at a dynamic position
    logic [2:0] byte_sel;
    logic [7:0] extracted_byte;
    assign byte_sel = lfsr[9:7];
    assign extracted_byte = data[byte_sel*8 +: 8];

    // Packed bit assignment: build a value bit by bit
    logic [7:0] bit_built;
    always_ff @(posedge clk) begin
        if (rst) begin
            bit_built <= 8'd0;
        end else begin
            // Set bit at position shift_amt[2:0] to lfsr[31]
            bit_built[shift_amt[2:0]] <= lfsr[31];
        end
    end

    // Packed range assignment: insert a nibble at a dynamic position
    logic [31:0] range_insert;
    logic [2:0] nibble_pos;
    assign nibble_pos = lfsr[12:10];
    always_ff @(posedge clk) begin
        if (rst) begin
            range_insert <= 32'd0;
        end else begin
            range_insert[nibble_pos*4 +: 4] <= lfsr[3:0];
        end
    end

    // Accumulate checksum
    always_ff @(posedge clk) begin
        if (rst)
            result <= 32'd0;
        else
            result <= result ^ shifted ^ {24'd0, extracted_byte}
                     ^ {24'd0, bit_built} ^ range_insert;
    end
endmodule
