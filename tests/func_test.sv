// Test: user-defined functions, foreach loops, break/continue
module func_test(
    input  logic        clk, rst,
    output logic [31:0] result
);
    // Pure function with conditional return
    function logic [31:0] clamp(input logic [31:0] val, input logic [31:0] max_val);
        if (val > max_val)
            return max_val;
        else
            return val;
    endfunction

    // Function with for-loop accumulation
    function logic [7:0] popcount8(input logic [7:0] val);
        logic [7:0] cnt;
        cnt = 8'd0;
        for (int i = 0; i < 8; i = i + 1) begin
            if (val[i])
                cnt = cnt + 8'd1;
        end
        return cnt;
    endfunction

    // Simple function — arithmetic
    function logic [31:0] triple(input logic [31:0] x);
        return x + x + x;
    endfunction

    logic [31:0] counter;
    logic [31:0] clamped;
    logic [7:0]  bits_set;
    logic [31:0] tripled;

    always_ff @(posedge clk) begin
        if (rst)
            counter <= 32'd0;
        else
            counter <= counter + 32'd100;
    end

    assign clamped  = clamp(counter, 32'd500);
    assign bits_set = popcount8(counter[7:0]);
    assign tripled  = triple(counter);

    // Foreach test: accumulate array elements
    logic [7:0] arr [0:3];
    always_ff @(posedge clk) begin
        if (rst) begin
            foreach (arr[i])
                arr[i] <= 8'd0;
        end else begin
            foreach (arr[i])
                arr[i] <= arr[i] + 8'(i) + 8'd1;
        end
    end

    // Checksum output combining function results and foreach array
    assign result = clamped ^ {24'd0, bits_set} ^ tripled ^ {arr[3], arr[2], arr[1], arr[0]};
endmodule
