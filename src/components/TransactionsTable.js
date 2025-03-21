import React from "react";

const TransactionsTable = () => {
  return (
    <div>
      <h2>Transaction Table</h2>
      <table border="1">
        <thead>
          <tr>
            <th>ID</th>
            <th>Amount</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>1</td>
            <td>$100</td>
            <td>Approved</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
};

export default TransactionsTable; // ✅ Ensure it's exported
