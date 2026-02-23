// CRC-32 with Ethernet polynomial — exercises XOR, shifts, conditionals, LFSR
module crc32 (
    input  logic        clk,
    input  logic        rst,
    output logic [31:0] result
);
    localparam [31:0] POLY = 32'h04C11DB7;  // CRC-32 Ethernet polynomial

    // LFSR for self-stimulus (8-bit data bytes)
    logic [31:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 32'hDEADBEEF;
        else
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    // CRC register
    logic [31:0] crc;
    logic [7:0]  data_byte;
    assign data_byte = lfsr[7:0];

    // CRC-32 computation: process one byte per cycle
    // Standard bit-by-bit CRC with XOR feedback
    logic [31:0] crc_next;
    always_comb begin
        crc_next = crc;
        for (int i = 0; i < 8; i = i + 1) begin
            if ((crc_next[31] ^ data_byte[i]) == 1'b1)
                crc_next = (crc_next << 1) ^ POLY;
            else
                crc_next = crc_next << 1;
        end
    end

    always_ff @(posedge clk) begin
        if (rst)
            crc <= 32'hFFFFFFFF;
        else
            crc <= crc_next;
    end

    // Accumulate CRC values
    logic [31:0] checksum;
    logic [7:0]  cycle_count;
    always_ff @(posedge clk) begin
        if (rst) begin
            checksum <= 32'd0;
            cycle_count <= 8'd0;
        end else begin
            checksum <= checksum ^ crc;
            cycle_count <= cycle_count + 8'd1;
        end
    end

    assign result = checksum;
endmodule
