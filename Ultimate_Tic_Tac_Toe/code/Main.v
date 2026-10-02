
`include"MB.v"

module TTT(
    input wire clk,
    input wire reset,
    input wire Up,
    input wire Down,
    input wire Left,
    input wire Right,
    input wire sel,
    output reg [5:0]row,
    output reg [11:0]col,
    output reg [1:0]Win
);

    reg [17:0]Tout;
    reg [1:0]T[8:0];
    reg [1:0]x;
    reg [1:0]y;
    reg who;
    reg [3:0]player1[3:0];
    reg [3:0]player2[3:0];
    reg [1:0]direct;
    reg [3:0]templayer[3:0];
    wire [5:0]Row;
    wire [11:0]Col;
    
    integer i , j , k , check;

    always @(posedge clk) begin // ­«¸m©M²¾°Ê
        if (reset) begin
            x <= 2'd1;
            y <= 2'd1;
            j = 0;
        end 
        else if(j == 0 && Win == 0) begin
            if (Up) begin
                j = 1;
                if (y < 2'd2) begin
                    y <= y + 1;
                end else begin
                    y <= 2'd0;
                end
            end 
            else if (Down) begin
                j = 1;
                if (y > 2'd0) begin
                    y <= y - 1;
                end else begin
                    y <= 2'd2;
                end
            end 
            else if (Left) begin
                j = 1;
                if (x > 2'd0) begin
                    x <= x - 1;
                end else begin
                    x <= 2'd2;
                end
            end 
            else if (Right) begin
                j = 1;
                if (x < 2'd2) begin
                    x <= x + 1;
                end else begin
                    x <= 2'd0;
                end
            end
        end
        else if(!Up && !Down && !Left && !Right && j == 1) begin
            j = 0;
        end
    end
    
    always @(posedge clk) begin
        if(reset) begin
            for(i = 0 ; i < 4 ; i = i + 1) begin
                player1[i] <= 4'd15;
                player2[i] <= 4'd15;
                templayer[i] <= 4'd15;
            end
            for (i = 0; i < 18; i = i + 1) begin
                Tout[i] <= 1'd0;
            end
            for (i = 0; i < 9; i = i + 1) begin
                T[i] <= 2'd0;
            end
            who <= 0;
            Win <= 2'd0;
            k = 0;
            check = 0;
        end
        else if(sel && k == 0) begin
            if (T[3 * y + x] == 2'd0 && Win == 2'd0) begin
                if (who == 1'd0) begin
                    T[3 * y + x] <= 2'd1;
                    player1[0] <= 3 * y + x;
                    who <= 1'd1;
                    templayer[2][3:0] <= player2[2][3:0];
                    templayer[1][3:0] <= player2[1][3:0];
                    templayer[0][3:0] <= player2[0][3:0];
                end else begin
                    T[3 * y + x] <= 2'd2;
                    player2[0] <= 3 * y + x;
                    who <= 1'd0;
                    templayer[2][3:0] <= player1[2][3:0];
                    templayer[1][3:0] <= player1[1][3:0];
                    templayer[0][3:0] <= player1[0][3:0];
                end
                k = 1;
                check = 0;
            end
        end
        else if (!sel && k == 1) begin
            if(check == 0) begin
                if(T[0] == T[1] && T[1] == T[2] && T[0] != 2'd0) begin
                    Win <= T[0];
                end
                else if(T[3] == T[4] && T[4] == T[5] && T[3] != 2'd0) begin
                    Win <= T[3];
                end
                else if(T[6] == T[7] && T[7] == T[8] && T[6] != 2'd0) begin
                    Win <= T[6];
                end
                else if(T[0] == T[3] && T[3] == T[6] && T[0] != 2'd0) begin
                    Win <= T[0];
                end
                else if(T[1] == T[4] && T[4] == T[7] && T[1] != 2'd0) begin
                    Win <= T[1];
                end
                else if(T[2] == T[5] && T[5] == T[8] && T[2] != 2'd0) begin
                    Win <= T[2];
                end
                else if(T[0] == T[4] && T[4] == T[8] && T[0] != 2'd0) begin
                    Win <= T[0];
                end
                else if(T[2] == T[4] && T[4] == T[6] && T[2] != 2'd0) begin
                    Win <= T[2];
                end
                check = 1;
            end
            else begin
                if(Win == 0) begin
                    if(who == 1'd1) begin
                        if(player2[2] != 4'd15) begin
                            T[player2[2]] <= 2'd0;
                        end
                        player2[3][3:0] <= templayer[2][3:0];
                        player2[2][3:0] <= templayer[1][3:0];
                        player2[1][3:0] <= templayer[0][3:0];
                    end
                    else begin
                        if(player1[2] != 4'd15) begin
                                T[player1[2]] <= 2'd0;
                        end
                        player1[3][3:0] <= templayer[2][3:0];
                        player1[2][3:0] <= templayer[1][3:0];
                        player1[1][3:0] <= templayer[0][3:0];
                    end
                    
                    k = 0;
                end
            end
        end
        Tout[1:0] <= T[0];
        Tout[3:2] <= T[1];
        Tout[5:4] <= T[2];
        Tout[7:6] <= T[3];
        Tout[9:8] <= T[4];
        Tout[11:10] <= T[5];
        Tout[13:12] <= T[6];
        Tout[15:14] <= T[7];
        Tout[17:16] <= T[8];
        
        row[5:0] <= Row[5:0];
        col[11:0] <= Col[11:0];
    end

    led_matrix LedM(
        .clk(clk),
        .reset(reset),
        .sel(sel),
        .p(3 * y + x),
        .Tout(Tout),
        .row(Row),
        .col(Col)
    );

 endmodule   