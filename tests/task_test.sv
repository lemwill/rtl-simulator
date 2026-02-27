module task_test(input logic clk, input logic rst, output logic [31:0] result);
    logic [31:0] counter;
    logic [31:0] acc;

    task automatic add_to_acc(input logic [31:0] val);
        acc <= acc + val;
    endtask

    always_ff @(posedge clk) begin
        if (rst) begin
            counter <= 32'd0;
            acc <= 32'd0;
            result <= 32'd0;
        end else begin
            counter <= counter + 32'd1;
            add_to_acc(counter);
            result <= acc;
        end
    end
endmodule
