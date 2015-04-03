function Gradient_Descent_regularized_coef(train_X,train_Y,test_X,test_Y)

% calculate w and b
[train_X_n,train_X_f]=size(train_X);
[test_X_n,test_X_f]=size(test_X);

lambda=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5];
step_size=[0.001,0.01,0.05,0.1,0.5];
T=50;


train_re_CE_all=zeros(length(step_size),length(lambda));

test_re_CE_all=zeros(length(step_size),length(lambda));

% regularized model
for s=1:length(step_size)
    train_re_CE=zeros(T,length(lambda));     %regularized corss-entropy error
    test_re_CE=zeros(T,length(lambda));     %regularized corss-entropy error
    step=step_size(s);
    
    for l=1:length(lambda)
        w=zeros(train_X_f,T+1);
        b=[0.1,zeros(1,T)];
        
        for t=1:T
            
            sum_w=zeros(train_X_f,1);
            sum_b=0;
            
            for n=1:train_X_n
                
                sigmoid=1/(1+exp(-w(:,t)'*train_X(n,:)'-b(t)));
                
                sum_b=sum_b+sigmoid-train_Y(n);
                sum_w=sum_w+(sigmoid-train_Y(n))*train_X(n,:)';
                
            end
            
            w(:,t+1)=w(:,t)-step*sum_w-2*lambda(l)*step*w(:,t);
            b(t+1)=b(t)-step*sum_b; 
           
            % CE for train data
            
            for n=1:train_X_n
                
                sigmoid=1/(1+exp(-w(:,t+1)'*train_X(n,:)'-b(t+1)));
                if sigmoid<10^(-16)
                    sigmoid=10^(-16);
                elseif sigmoid>(1-10^(-16))
                    sigmoid=1-10^(-16);
                end
                train_re_CE(t,l)=train_re_CE(t,l)-train_Y(n)*log(sigmoid)-(1-train_Y(n))*log(1-sigmoid);
                
            end
            
            train_re_CE(t,l)=train_re_CE(t,l)+lambda(l)*norm(w(:,t+1))^2;
            
            % CE for test data
            
            for n=1:test_X_n
                
                sigmoid=1/(1+exp(-(w(:,t+1)'*test_X(n,:)'+b(t+1))));
                if sigmoid<10^(-16)
                    sigmoid=10^(-16);
                elseif sigmoid>(1-10^(-16))
                    sigmoid=1-10^(-16);
                end
                test_re_CE(t,l)=test_re_CE(t,l)-test_Y(n)*log(sigmoid)-(1-test_Y(n))*log(1-sigmoid);
                
            end
            
            test_re_CE(t,l)=test_re_CE(t,l)+lambda(l)*norm(w(:,t+1))^2;
            
        end
        
        
    end
    
    
    train_re_CE_all(s,:)=train_re_CE(T,:);
    test_re_CE_all(s,:)=test_re_CE(T,:);
    
    % plot cross-entropy function value
    figure
    plot(lambda,train_re_CE_all(s,:),'-.b',lambda,test_re_CE_all(s,:),'-ro')
    str=sprintf('Regularized Cross-Entropy, step size= %d', step_size(s));
    title(str);
    xlabel('Regularization Coefficient');
    ylabel('Regularized Cross-Entropy value after 50 iterations');
    legend('train','test');
end

