clear; clc; close all;
figure();
% estimation
est = csvread("log.csv"); hold on;
plot3(est(:,1),est(:,3),-est(:,2));
axis equal;
grid on;
% for i = 1:size(est,1)
%    if norm([est(i,1) - est(1,1), est(i,3) - est(1,3)]) > 0.3
%        est_vec = [est(i,1) - est(1,1), est(i,3) - est(1,3)];
%        break;
%    end
% end
% % ground truth
gt = csvread("mav0/state_groundtruth_estimate0/gt.csv");
plot3(gt(:,2),gt(:,3),gt(:,4));
% for i = 1:size(gt,1)
%    if norm([gt(i,2) - gt(1,2), gt(i,3) - gt(1,3)]) > norm(est_vec)
%        gt_vec = [gt(i,2) - gt(1,2), gt(i,3) - gt(1,3)];
%        break;
%    end
% end
% theta = atan2(est_vec(2),est_vec(1)) - atan2(gt_vec(2),gt_vec(1));
% rot = [cos(theta), -sin(theta);sin(theta),cos(theta)];
% pts = rot * [gt(:,2)';gt(:,3)'];
% plot(pts(1,:),pts(2,:));
% axis equal;
% grid on;