function [err] = MSE(actual, predicted)
    notNanIdxs = find(~isnan(actual) & ~isnan(predicted));
    actual = actual(notNanIdxs); predicted = predicted(notNanIdxs);
    err = norm(actual-predicted, 2).^2 ./ norm(actual,2).^2;
end