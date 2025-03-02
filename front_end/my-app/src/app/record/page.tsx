"use client";
import { useState, useEffect } from "react";
import Link from "next/link";

export default function RecordPage() {
  interface RecordType {
    id: number;
    filename: string;
    time: string;
    person_count: number;
  }
  const [records, setRecords] = useState<RecordType[]>([]);
  const [filteredRecords, setFilteredRecords] = useState<RecordType[]>([]);
  const [pageNumber, setPageNumber] = useState(1);
  const [loading, setLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState("");

  useEffect(() => {
    fetchRecords(pageNumber);
  }, [pageNumber]);

  const fetchRecords = async (page:number) => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/record?page_number=${page}`, {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      });
      const data = await response.json();
      console.log(data, `http://localhost:8000/record?page_number=${page}`)
      setRecords(data);
      setFilteredRecords(data); // Set filtered records to all records initially
    } catch (error) {
      console.error("Error fetching records:", error);
    } finally {
      setLoading(false);
    }
    console.log(page)
  };

  const handleSearch = async (event:React.ChangeEvent<HTMLInputElement> ) => {
    const term = event.target.value.trim().toLowerCase();
    setSearchTerm(term);
    
    if (term === "") {
        setFilteredRecords(records);  // Reset to current page records
        return;
    }

    try {
        const response = await fetch(`http://localhost:8000/record?id=${term}`);
        
        if (!response.ok) {
            throw new Error("Failed to fetch records");
        }

        const data = await response.json();
        setFilteredRecords(data);
    } catch (error) {
        console.error("Search error:", error);
        setFilteredRecords([]);  // Fallback to empty state on error
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4 flex items-center gap-6">
        Detection Records
        <Link href="/" className="text-sm text-white hover:text-gray-300 hover:underline">
          Home
        </Link>
        <Link href="/detection" className="text-sm text-white hover:text-gray-300 hover:underline">
          Image Detection
        </Link>
      </h1>

      <div className="mb-4">
        <input
          type="text"
          placeholder="Search records by id..."
          value={searchTerm}
          onChange={handleSearch}
          className="w-full p-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
      </div>

      {loading ? (
        <p>Loading...</p>
      ) : (
        <table className="min-w-full bg-white border border-gray-200 shadow-md rounded-lg">
          <thead>
            <tr className="bg-gray-200 text-gray-700">
              <th className="p-3 text-left">ID</th>
              <th className="p-3 text-left">Filename</th>
              <th className="p-3 text-left">Timestamp</th>
              <th className="p-3 text-left">Person Count</th>
            </tr>
          </thead>
          <tbody>
            {filteredRecords.length > 0 ? (
              filteredRecords.map((record) => (
                <tr key={record.id} className="border-b hover:bg-gray-100 text-black">
                  <td className="p-3">{record.id}</td>
                  <td className="p-3">{record.filename}</td>
                  <td className="p-3">{new Date(record.time).toLocaleString()}</td>
                  <td className="p-3">{record.person_count}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={4} className="p-3 text-center">
                  No records found.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      )}

      <div className="mt-4 flex justify-center space-x-4">
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
          disabled={pageNumber === 1}
          onClick={() => setPageNumber(pageNumber - 1)}
        >
          Previous
        </button>
        <span className="px-4 py-2">Page {pageNumber}</span>
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300"
          onClick={() => setPageNumber(pageNumber + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
}
