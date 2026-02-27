// 2D Memory Test: 4x4 byte memory with read/write and checksum
module mem2d_test (
    input  logic        clk, rst,
    output logic [31:0] result
);
    // 2D unpacked array: 4 rows x 4 columns of bytes
    logic [7:0] mem [0:3][0:3];

    logic [3:0]  cycle_count;
    logic [1:0]  wr_row, wr_col, rd_row, rd_col;
    logic [7:0]  wr_data, rd_data;
    logic [31:0] checksum;

    // LFSR for pseudo-random stimulus
    logic [15:0] lfsr;

    assign wr_row  = lfsr[1:0];
    assign wr_col  = lfsr[3:2];
    assign wr_data = lfsr[11:4];
    assign rd_row  = lfsr[13:12];
    assign rd_col  = lfsr[15:14];

    // Dynamic array read
    assign rd_data = mem[rd_row][rd_col];

    always_ff @(posedge clk) begin
        if (rst) begin
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    mem[i][j] <= 8'd0;
            lfsr <= 16'hACE1;
            cycle_count <= 4'd0;
            checksum <= 32'd0;
            result <= 32'd0;
        end else begin
            // LFSR shift
            lfsr <= {lfsr[14:0], lfsr[15] ^ lfsr[13] ^ lfsr[12] ^ lfsr[10]};

            // Dynamic array write
            mem[wr_row][wr_col] <= wr_data;

            // Accumulate read data into checksum
            checksum <= checksum ^ {24'd0, rd_data};

            cycle_count <= cycle_count + 4'd1;
            result <= checksum;
        end
    end
endmodule
