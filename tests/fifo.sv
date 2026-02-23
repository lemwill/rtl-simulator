// Parameterized synchronous FIFO — exercises $clog2, arrays, counters
module fifo #(
    parameter DEPTH = 8,
    parameter WIDTH = 32
)(
    input  logic             clk,
    input  logic             rst,
    output logic [WIDTH-1:0] result
);
    localparam ADDR_W = $clog2(DEPTH);

    // Storage
    logic [WIDTH-1:0] mem [0:DEPTH-1];

    // Pointers and status
    logic [ADDR_W:0] wr_ptr, rd_ptr;
    logic full, empty;
    assign full  = (wr_ptr[ADDR_W] != rd_ptr[ADDR_W]) &&
                   (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0]);
    assign empty = (wr_ptr == rd_ptr);

    // LFSR for self-stimulus
    logic [31:0] lfsr;
    always_ff @(posedge clk) begin
        if (rst)
            lfsr <= 32'hCAFEBABE;
        else
            lfsr <= {lfsr[30:0], lfsr[31] ^ lfsr[21] ^ lfsr[1] ^ lfsr[0]};
    end

    // Write logic
    logic wr_en;
    assign wr_en = lfsr[0] & ~full;
    always_ff @(posedge clk) begin
        if (rst) begin
            wr_ptr <= '0;
        end else if (wr_en) begin
            mem[wr_ptr[ADDR_W-1:0]] <= lfsr;
            wr_ptr <= wr_ptr + 1;
        end
    end

    // Read logic
    logic rd_en;
    assign rd_en = lfsr[1] & ~empty;
    logic [WIDTH-1:0] rd_data;
    assign rd_data = mem[rd_ptr[ADDR_W-1:0]];
    always_ff @(posedge clk) begin
        if (rst) begin
            rd_ptr <= '0;
        end else if (rd_en) begin
            rd_ptr <= rd_ptr + 1;
        end
    end

    // Checksum accumulator
    logic [WIDTH-1:0] checksum;
    always_ff @(posedge clk) begin
        if (rst)
            checksum <= '0;
        else if (rd_en)
            checksum <= checksum ^ rd_data;
    end

    assign result = checksum;
endmodule
