close all
num = xlsread('rewards.xlsx');
len = 120;
step = num(1:len,1);
rbefore = num(1:len,2:4:22);
rafter = num(1:len,3:4:23);

% figure(1)
hold on
p2 = semilogy(step, rbefore,'Color',[0.8 0.8 0.8],'LineWidth',0.2);
p1 = plot(step, smoothdata(rbefore),'LineWidth',1.5);
legend([p1(1), p1(2),p1(3),p1(4),p1(5),p1(6), p2(1)], {'SA+MAML 1','SA+MAML 2','SA+MAML 3','MAML 1','MAML 2', 'MAML 3', 'Raw data'})
grid on
xlabel('Iteration')
ylabel('Reward')
ylim([-140 0])
xlim([0 120])